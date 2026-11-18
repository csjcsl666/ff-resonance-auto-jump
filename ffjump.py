#!/usr/bin/env python3
"""FF 跳绳小游戏自动跳绳

  ffjump.py run            待命；NumPad 0 开启/关闭自动跳绳（唯一的快捷键）
  ffjump.py run --dry-run  只检测记录，不发 F
  ffjump.py replay CLIPS   用已保存的 ROI 片段离线回放触发逻辑（不发键）
  ffjump.py info           环境调查

触发依据（来自实测）：玩家头顶弹出半透明「！」气泡 -> 以弹出时刻为基准、延迟 offset_ms 后按 F。
反馈依据：计数数字变化=成功；整屏渐黑=游戏结束；气泡消失时刻、内核按键回显、窗口焦点用于失败归因。
"""
import argparse, collections, fcntl, json, logging, os, queue, random, re, select, signal, struct
import subprocess, sys, threading, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CFG_PATH, TUNE_PATH = HERE / "ffjump.json", HERE / "tuning.json"
KEY_F, KEY_KP0 = 33, 82
now = time.perf_counter

DEFAULTS = {
    "window_class": "steam_app_3259780",
    "ref_w": 2048,
    "template": "templates/bubble_A.png",
    "template_k": 1.25,                 # 模板采集时 物理像素/参考像素
    "roi": [880, 330, 290, 470],        # 捕获区(参考坐标)：含头顶气泡搜索区与计数数字
    "bubble_ref": [880, 440, 290, 360], # 气泡搜索区
    "counter_ref": [968, 371, 112, 48], # 计数数字区
    "thr_off": 0.55, "on_frames": 1, "off_frames": 2,
    "counter_bright": 215, "counter_change_frac": 0.6,
    "digit_px_range": [80, 2500],        # 计数数字区的合理亮像素数（实测 "0" 约 365）
    "success_window_s": 0.7,            # 按 F 后多久内计数应变化
    "cue_period_s": 0.90,               # 初值；运行中按最近提示间隔的中位数自适应
    "dither_ms": [],                    # 正式运行不抖动：延迟是确定的
    "pop_midpoint": False,              # False=以检测到的帧时刻为基准（与已验证的 110ms 基线一致）
    "scale_with_visible": True,         # 按最近气泡可见时长缩放延迟（只会缩短，不会超过基线）
    "visible_ref_ms": 255.0, "scale_min": 0.4,
    "hint_ref": [1068, 750, 52, 49],    # 右下角操作提示图标（手柄 A / 键盘 F 切换处）
    "first_cue_offset_ms": None,        # 新局第一个提示用不同延迟（默认关闭，需显式指定）
    "dark_ratio": 0.35, "dark_frames": 3,
    "focus_check": True,
}
BASELINE_PATH = HERE / "baseline.json"
TUNE_DEFAULTS = {"offset_ms": 110.0, "hold_ms": 80.0, "thr_on": 0.70, "bins": {}, "runs": 0}
LIMITS = {"offset_ms": (30.0, 250.0), "hold_ms": (50.0, 150.0), "thr_on": (0.58, 0.85)}


def load_json(p, default):
    d = json.loads(p.read_text()) if p.exists() else {}
    return {**default, **d}


def load_tune(use_baseline=False):
    """基线(baseline.json，程序不写) < 学习结果(tuning.json)。--baseline 时忽略学习结果"""
    t = {**TUNE_DEFAULTS}
    if BASELINE_PATH.exists():
        t.update({k: v for k, v in json.loads(BASELINE_PATH.read_text()).items() if not k.startswith("_")})
    if not use_baseline and TUNE_PATH.exists():
        t.update(json.loads(TUNE_PATH.read_text()))
    t["bins"] = dict(t.get("bins", {}))
    return t


def load_cfg():
    return load_json(CFG_PATH, DEFAULTS)


def sh_json(*cmd):
    return json.loads(subprocess.check_output(cmd, timeout=2))


# ================================================================ 窗口几何
class Geometry:
    def __init__(self, cfg):
        win = next((c for c in sh_json("hyprctl", "clients", "-j") if c["class"] == cfg["window_class"]), None)
        if win is None:
            raise RuntimeError(f"找不到游戏窗口 class={cfg['window_class']}")
        mon = next(m for m in sh_json("hyprctl", "monitors", "-j") if m["id"] == win["monitor"])
        self.win_x, self.win_y = win["at"]
        self.win_w, self.win_h = win["size"]
        self.mon_scale = mon["scale"]
        self.k = self.mon_scale * self.win_w / cfg["ref_w"]        # 物理 px / 参考 px
        f = self.win_w / cfg["ref_w"]                              # 参考 -> 逻辑
        rx, ry, rw, rh = cfg["roi"]
        self.lx, self.ly = int(round(self.win_x + rx * f)), int(round(self.win_y + ry * f))
        self.lw, self.lh = int(round(rw * f)), int(round(rh * f))

    key = property(lambda s: (s.win_x, s.win_y, s.win_w, s.win_h, s.mon_scale))
    region = lambda s: f"{s.lx},{s.ly} {s.lw}x{s.lh}"

    def sub(self, cfg, ref_rect):
        """参考坐标矩形 -> 捕获帧(物理像素)内的 (y0,y1,x0,x1)"""
        rx, ry = cfg["roi"][:2]
        x, y, w, h = ref_rect
        return (int(round((y - ry) * self.k)), int(round((y - ry + h) * self.k)),
                int(round((x - rx) * self.k)), int(round((x - rx + w) * self.k)))


# ================================================================ 持续捕获
class Capture:
    """wf-recorder rawvideo 管道，持续输出 ROI 帧（-D：不依赖画面变化）"""

    def __init__(self, region, log, extra=None):
        self.region, self.log, self.extra = region, log, extra or []
        self.cv = threading.Condition()
        self.idx, self.ts, self.buf, self.size, self.dead, self.proc = 0, 0.0, None, None, False, None

    def start(self):
        cmd = ["wf-recorder"] + (["-g", self.region] if self.region else []) + self.extra + [
            "-D", "-y", "-c", "rawvideo", "-x", "bgr0", "-m", "rawvideo", "-f", "/dev/stdout"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        try:
            fcntl.fcntl(self.proc.stdout.fileno(), 1031, 1 << 20)  # F_SETPIPE_SZ
        except OSError:
            pass
        sized = threading.Event()
        threading.Thread(target=self._err, args=(sized,), daemon=True).start()
        if not sized.wait(5):
            raise RuntimeError("wf-recorder 未报告帧尺寸")
        threading.Thread(target=self._read, daemon=True).start()

    def _err(self, sized):
        for line in self.proc.stderr:
            s = line.decode(errors="replace").rstrip()
            m = re.search(r"Video: rawvideo.* (\d+)x(\d+)", s)
            if m and self.size is None:
                self.size = (int(m[1]), int(m[2])); sized.set()
            elif s and "filter" not in s and "Changing video" not in s:
                self.log.debug("wf-recorder: %s", s)

    def _read(self):
        w, h = self.size
        n, f = w * h * 4, self.proc.stdout
        while True:
            buf = bytearray(n); mv, got = memoryview(buf), 0
            while got < n:
                r = f.readinto(mv[got:])
                if not r:
                    with self.cv:
                        self.dead = True; self.cv.notify_all()
                    return
                got += r
            ts = now()
            with self.cv:
                self.idx += 1; self.ts, self.buf = ts, buf
                self.cv.notify_all()

    def wait(self, last_idx, timeout):
        with self.cv:
            if self.idx == last_idx and not self.dead:
                self.cv.wait(timeout)
            return self.idx, self.ts, self.buf, self.dead

    def wake(self):
        with self.cv:
            self.cv.notify_all()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(2)
            except subprocess.TimeoutExpired:
                self.proc.terminate()


def frame_to_gray(buf, size):
    w, h = size
    return cv2.cvtColor(np.frombuffer(buf, np.uint8).reshape(h, w, 4), cv2.COLOR_BGRA2GRAY)


def match(gray, tpl):
    if gray.shape[0] < tpl.shape[0] or gray.shape[1] < tpl.shape[1]:
        return 0.0, (0, 0)
    _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED))
    return (0.0 if mx != mx else float(mx)), loc


# ================================================================ 按键注入
class UInputKey:
    """/dev/uinput 虚拟键盘。每次按键后读自己设备的 evdev 回显，确认内核已发出事件"""

    def __init__(self, log):
        self.log, self.down, self.last_release_lag_ms = log, False, None
        self.fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
        fcntl.ioctl(self.fd, 0x40045564, 1)          # UI_SET_EVBIT EV_KEY
        fcntl.ioctl(self.fd, 0x40045565, KEY_F)      # UI_SET_KEYBIT
        os.write(self.fd, b"ffjump virtual keyboard".ljust(80, b"\0")
                 + struct.pack("<HHHHi", 3, 0x1234, 0x5678, 1, 0) + bytes(256 * 4))
        fcntl.ioctl(self.fd, 0x5501)                 # UI_DEV_CREATE
        time.sleep(1.0)
        self.echo_fd = None
        for blk in open("/proc/bus/input/devices").read().split("\n\n"):
            if "ffjump virtual keyboard" in blk:
                node = re.search(r"event\d+", blk)
                if node:
                    try:
                        self.echo_fd = os.open("/dev/input/" + node[0], os.O_RDONLY | os.O_NONBLOCK)
                        fcntl.ioctl(self.echo_fd, 0x400445a0, struct.pack("i", time.CLOCK_MONOTONIC))
                    except OSError as e:
                        log.warning("无法读取回显设备: %s", e)
        log.info("uinput 虚拟键盘已创建，回显校验=%s", "开" if self.echo_fd is not None else "关")

    def _emit(self, value):
        t = time.time()
        sec, usec = int(t), int((t % 1) * 1e6)
        os.write(self.fd, struct.pack("<qqHHi", sec, usec, 1, KEY_F, value) + struct.pack("<qqHHi", sec, usec, 0, 0, 0))

    def _echo(self, value, deadline=0.02):
        """等待内核回显，返回内核事件时间戳(perf 时基，秒)；无回显返回 False，未启用回显返回 None"""
        if self.echo_fd is None:
            return None
        end = now() + deadline
        while now() < end:
            if not select.select([self.echo_fd], [], [], max(0, end - now()))[0]:
                break
            data = os.read(self.echo_fd, 24 * 16)
            for i in range(0, len(data) - 23, 24):
                s, us, typ, code, val = struct.unpack_from("<qqHHi", data, i)
                if typ == 1 and code == KEY_F and val == value:
                    return s + us / 1e6          # 设备按 CLOCK_MONOTONIC 打时间戳(EVIOCSCLOCKID)，与 perf_counter 同时基
        return False

    def press(self):
        t0 = now()
        try:
            self._emit(1); self.down = True
        except OSError as e:
            return {"ok": False, "err": str(e), "t_write": t0, "t_done": now()}
        t1 = now()
        echo = self._echo(1)
        return {"ok": echo is not False, "echo": echo is not False, "kernel_lag_ms": round((echo - t0) * 1000, 3) if echo else None,
                "t_write": t0, "t_done": t1, "err": "无内核回显" if echo is False else None}

    def release(self):
        if not self.down:
            return None
        t0 = now()
        try:
            self._emit(0)
        finally:
            self.down = False
        e = self._echo(0)
        self.last_release_lag_ms = round((e - t0) * 1000, 3) if e else None
        return e is not False

    def close(self):
        try:
            self.release()
            fcntl.ioctl(self.fd, 0x5502)             # UI_DEV_DESTROY（内核同时释放所有按键）
        finally:
            os.close(self.fd)


class FakeKey:
    """dry-run / replay 用：不发任何键"""
    down = False

    def __init__(self, clock=now):
        self.clock = clock

    def press(self):
        t = self.clock(); return {"ok": True, "echo": None, "t_write": t, "t_done": t, "err": None}

    def release(self):
        return None

    def close(self):
        pass


# ================================================================ NumPad 0 监听
class ToggleKey(threading.Thread):
    """只读 evdev 监听 NumPad 0（不 grab，不依赖焦点，游戏在前台也有效），只保留这一个键"""

    def __init__(self, on_toggle, log):
        super().__init__(daemon=True)
        self.on_toggle, self.log, self.fds = on_toggle, log, []

    def run(self):
        for blk in open("/proc/bus/input/devices").read().split("\n\n"):
            h = next((l for l in blk.splitlines() if l.startswith("H: Handlers=")), "")
            key = next((l for l in blk.splitlines() if l.startswith("B: KEY=")), "")
            name = next((l for l in blk.splitlines() if l.startswith("N: Name=")), "")
            if "kbd" not in h or not key or "ffjump" in name or "ydotoold" in name:
                continue
            bits = int("".join(w.zfill(16) for w in key.split("=")[1].split()), 16)
            if (bits >> KEY_KP0) & 1 and (bits >> KEY_F) & 1:
                path = "/dev/input/" + next(t for t in h.split() if t.startswith("event"))
                try:
                    self.fds.append(os.open(path, os.O_RDONLY | os.O_NONBLOCK))
                    self.log.info("NumPad 0 监听(只读): %s", name[9:].strip('"'))
                except OSError as e:
                    self.log.warning("无法读取 %s: %s", path, e)
        if not self.fds:
            self.log.error("没有可读的带小键盘的键盘设备，NumPad 0 无法使用"); return
        while True:
            for fd in select.select(self.fds, [], [])[0]:
                try:
                    data = os.read(fd, 24 * 64)
                except OSError:
                    continue
                for i in range(0, len(data) - 23, 24):
                    _, _, typ, code, val = struct.unpack_from("<qqHHi", data, i)
                    if typ == 1 and code == KEY_KP0 and val == 1:
                        self.on_toggle()


# ================================================================ 触发/反馈引擎
def wilson_lb(s, n, z=1.28):
    if n == 0:
        return 0.0
    p = s / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


class Engine:
    """帧驱动的状态机；时钟和按键均由外部注入，因此同一份逻辑可在线运行也可离线回放"""

    def __init__(self, cfg, tune, tpl, key, emit, focused=lambda: True, have_counter=True):
        self.cfg, self.tune, self.tpl, self.key, self.emit = cfg, tune, tpl, key, emit
        self.focused, self.have_counter = focused, have_counter
        self.enabled = False
        self.state = "OFF"                    # OFF / ARMED / SCHED / HELD
        self.present = False
        self._on = self._off = 0
        self.press_at = self.release_at = 0.0
        self.cur = None                       # 已排程但未按下的 J
        self.pending = []                     # 已按下、等待结果的 J
        self.jid = 0
        self.last_pop = None; self.last_press = None; self.last_success = None
        self.gap_max = 0.0; self.miss_flagged = False
        self.ctr_ref = None; self.ctr_cool = 0.0
        self.bright_hist = collections.deque(maxlen=300); self._dark = 0
        self.consec = collections.Counter()
        self.stats = collections.Counter()
        self.on_dump = lambda reason, t: None
        self.prev_ts = None; self.last_score = 0.0
        self.period = cfg["cue_period_s"]; self.pop_hist = collections.deque(maxlen=9)
        self.vis_hist = collections.deque(maxlen=20); self.ivl_hist = collections.deque(maxlen=20)
        self.dark_since_pop = False
        self.rng = random.Random(12345)
        self.last_interval = None
        self.hint_at_press = None; self.hint_until = 0.0; self.hint_press_ts = None; self._hint_now = None
        self.hint_tpl = {}
        for k in ("A", "F"):
            p = HERE / "templates" / f"hint_{k}.png"
            if p.exists():
                self.hint_tpl[k] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.int16)
        self.score_hist = collections.deque(maxlen=40)       # (ts, score)：轻量内存证据，失败时随结果写入日志
        self.blk = {"ivl": [], "vis": [], "ok": 0, "fail": 0, "off": [], "n": 0}

    # ---- 开关
    def enable(self, t):
        self.enabled, self.state = True, "ARMED"
        self.cur, self.pending, self.consec = None, [], collections.Counter()
        self.miss_flagged, self.gap_max, self.ctr_ref = False, 0.0, None
        self.last_success = None
        self._reset_tempo()
        self.emit("enabled", t, **self.params())

    def _reset_tempo(self):
        self.pop_hist.clear(); self.vis_hist.clear(); self.ivl_hist.clear()
        self.period = self.cfg["cue_period_s"]; self.last_interval = None; self.last_pop = None

    def disable(self, t, reason="user"):
        if self.key.down:
            self.key.release()                    # 立即松开
        self.enabled, self.state, self.cur = False, "OFF", None
        self.pending = []
        self.emit("disabled", t, reason=reason)

    def params(self):
        return {k: self.tune[k] for k in ("offset_ms", "hold_ms", "thr_on")}

    # ---- 帧输入
    def on_frame(self, ts, bub, ctr=None, bright=None, hint=None):
        score, loc = match(bub, self.tpl)
        thr = self.cfg["thr_off"] if self.present else self.tune["thr_on"]
        if score >= thr:
            self._on, self._off = self._on + 1, 0
        else:
            self._on, self._off = 0, self._off + 1
        if not self.present:
            self.gap_max = max(self.gap_max, score)
        self.score_hist.append((ts, score))
        self._hint_now = hint
        gap = (ts - self.prev_ts) if self.prev_ts is not None else None
        prev_score = self.last_score
        self.prev_ts, self.last_score = ts, score
        if not self.present and self._on >= self.cfg["on_frames"]:
            self.present = True; self._pop(ts, score, loc, gap, prev_score)
        elif self.present and self._off >= self.cfg["off_frames"]:
            self.present = False; self._vanish(ts)
        if hint is not None and self.hint_at_press == "A" and ts <= self.hint_until:
            if self._hint_label(hint) == "F":            # 按键时提示是手柄 A，之后变为键盘 F：游戏已处理到键盘输入
                ms = (ts - self.hint_press_ts) * 1000
                self.emit("hint_changed", ts, ms_after_press=round(ms, 1))
                if self.pending:
                    self.pending[-1]["hint_ms"] = ms
                self.hint_at_press = None
        if ctr is not None:
            self._counter(ts, ctr, bright)
        if bright is not None:
            self._bright(ts, bright)
        self.tick(ts)
        return score

    def _hint_label(self, img):
        """右下角操作提示图标分类：'A'(手柄) / 'F'(键盘) / None(未知或被遮挡)"""
        t = self.hint_tpl
        if img is None or "A" not in t or img.shape != t["A"].shape:
            return None
        x = img.astype(np.int16)
        dA, dF = float(np.abs(x - t["A"]).mean()), float(np.abs(x - t["F"]).mean())
        return "A" if dA < dF and dA < 18 else "F" if dF < dA and dF < 18 else None

    def _scaled_offset(self):
        """确定性延迟：基线 offset，且仅当最近 3 个气泡可见时长明显短于基线时按比例缩短（变速保护）"""
        center = self.tune["offset_ms"]
        if self.cfg["scale_with_visible"] and len(self.vis_hist) >= 3:
            v = float(np.median(list(self.vis_hist)[-3:]))
            scale = min(1.0, max(self.cfg["scale_min"], v / self.cfg["visible_ref_ms"]))
            return max(LIMITS["offset_ms"][0], center * scale), scale
        return center, 1.0

    def _pop(self, ts, score, loc, gap=None, prev_score=0.0):
        self.stats["pops"] += 1
        iv_ms = None
        first_cue = self.last_pop is None
        if self.last_pop is not None:
            iv = ts - self.last_pop
            if 0.2 < iv < 2.5:                        # 实测提示间隔（不依赖固定节拍；仅供保护逻辑与日志）
                iv_ms = iv * 1000
                self.last_interval = iv
                self.pop_hist.append(iv); self.ivl_hist.append(iv); self.blk["ivl"].append(iv * 1000)
                if len(self.pop_hist) >= 3:
                    self.period = float(np.median(list(self.pop_hist)[-5:]))
        self.last_pop, self.gap_max, self.miss_flagged, self.dark_since_pop = ts, 0.0, False, False
        est = ts - gap / 2 if self.cfg["pop_midpoint"] and gap is not None and 0 < gap < 0.12 else ts
        self.emit("pop", ts, score=round(score, 3), loc=list(map(int, loc)), gap_ms=round(gap * 1000, 1) if gap else None,
                  prev_score=round(prev_score, 3), interval_ms=round(iv_ms, 1) if iv_ms else None, period_s=round(self.period, 3),
                  first_cue=first_cue)
        if self.enabled and self.state == "ARMED":
            self.jid += 1
            off, scale = self._scaled_offset()
            fo = self.cfg.get("first_cue_offset_ms")
            if first_cue and fo:
                off = float(fo)                          # 仅新局第一个提示（显式指定才生效）
            dith = self.cfg["dither_ms"]
            if dith:
                off += self.rng.choice(dith)
            off = max(LIMITS["offset_ms"][0], min(LIMITS["offset_ms"][1], off))
            self.cur = {"id": self.jid, "pop_ts": ts, "pop_est": est, "offset_ms": off, "center_ms": self.tune["offset_ms"],
                        "scale": scale, "first_cue": first_cue, "pop_score": score, "pop_gap_ms": gap * 1000 if gap else None, "prev_score": prev_score,
                        "interval_ms": iv_ms}
            self.press_at = est + off / 1000
            self.state = "SCHED"

    def _vanish(self, ts):
        vis = (ts - self.last_pop) * 1000 if self.last_pop else None
        if vis:
            self.vis_hist.append(vis); self.blk["vis"].append(vis)
        self.emit("vanish", ts, visible_ms=round(vis, 1) if vis else None, pressed=bool(self.pending or self.cur))
        for j in self.pending:
            j.setdefault("vanish_ts", ts)
        if self.cur is not None:
            self.cur.setdefault("vanish_ts", ts)

    def _counter(self, ts, ctr, bright=None):
        m = ctr > self.cfg["counter_bright"]
        if self.ctr_ref is None:
            self.ctr_ref = m; return
        lo, hi = self.cfg["digit_px_range"]
        base = float(np.median(self.bright_hist)) if len(self.bright_hist) > 20 else None
        fading = self._dark > 0 or (bright is not None and base and bright < 0.8 * base)
        plausible = lo <= int(m.sum()) <= hi and lo <= int(self.ctr_ref.sum()) <= hi   # 新旧都像数字
        if fading or not plausible:
            self.ctr_ref = m                         # 淡入淡出/计数框出现消失：只更新参考，不算计数变化
            return
        frac = float((m != self.ctr_ref).sum() / max(1.0, (m.sum() + self.ctr_ref.sum()) / 2))
        if frac > self.cfg["counter_change_frac"] and ts > self.ctr_cool:
            self.ctr_cool = ts + 0.25
            self.ctr_ref = m
            self.emit("counter_change", ts, frac=round(frac, 2), bright_px=int(m.sum()))
            if self.enabled and self.pending:
                self._resolve(self.pending.pop(0), True, ts)
        elif frac < 0.1:
            self.ctr_ref = m

    def _bright(self, ts, b):
        base = float(np.median(self.bright_hist)) if len(self.bright_hist) > 20 else None
        if base and b < base * self.cfg["dark_ratio"]:
            self._dark += 1
            if self._dark == self.cfg["dark_frames"]:
                # 仅记录：开场淡入/淡出、失败后淡出都会触发，不能据此判断游戏结束，更不能控制开关
                self.emit("dark_screen", ts, brightness=round(b, 1), baseline=round(base, 1),
                          since_last_pop_s=round(ts - self.last_pop, 2) if self.last_pop else None,
                          last_press_ago_s=round(ts - self.last_press, 2) if self.last_press else None)
                self.on_dump("dark_screen", ts)
                self.dark_since_pop = True
                self._reset_tempo()                       # 转场后是新局/结算，节拍重新测量
        else:
            self._dark = 0
            self.bright_hist.append(b)

    # ---- 时钟
    def next_deadline(self):
        d = []
        if self.state == "SCHED": d.append(self.press_at)
        if self.state == "HELD": d.append(self.release_at)
        d += [j["deadline"] for j in self.pending]
        return min(d) if d else None

    def tick(self, t):
        if self.state == "HELD" and t >= self.release_at:
            echo = self.key.release()
            self.emit("release", t, echo=echo, kernel_lag_ms=getattr(self.key, "last_release_lag_ms", None))
            self.state = "ARMED"
        if self.state == "SCHED" and t >= self.press_at:
            self._do_press(t)
        for j in list(self.pending):
            if t >= j["deadline"] and self.have_counter:
                self.pending.remove(j); self._resolve(j, False, t)
        # 漏检：连招中超过约 1.9 个节拍没有任何提示
        if (self.enabled and self.last_success is not None and self.last_pop is not None and not self.miss_flagged
                and t - self.last_success < 4 and t - self.last_pop > 1.9 * max(self.period, self.last_interval or 0)):
            self._visual_miss(t)

    def _do_press(self, t):
        j = self.cur
        if not self.present:                       # 气泡在按键前已消失 -> 已错过
            self.emit("cancel", t, jump=j["id"], why="cue_gone_before_press")
            self.stats["cancel_gone"] += 1
            self.consec["late"] += 1; self.consec["early"] = 0
            self._bin(j, False)
            self._maybe_adjust(t)
            self.state, self.cur = "ARMED", None
            return
        if self.last_press is not None and t - self.last_press < 0.5 * min(self.period, self.last_interval or self.period):
            self.emit("cancel", t, jump=j["id"], why="repeat_guard")
            self.stats["repeat_blocked"] += 1
            self.state, self.cur = "ARMED", None
            return
        if self.cfg["focus_check"] and not self.focused():
            self.emit("cancel", t, jump=j["id"], why="not_focused")
            self.stats["inject_fail"] += 1
            self.state, self.cur = "ARMED", None
            return
        j["hint_at_press"] = self._hint_label(self._hint_now) if self._hint_now is not None else None
        r = self.key.press()
        j.update(press_ts=r["t_write"], inj_ok=r["ok"], echo=r.get("echo"),
                 deadline=r["t_write"] + self.cfg["success_window_s"])
        self.last_press = r["t_write"]
        self.stats["presses"] += 1
        self.emit("press", r["t_write"], jump=j["id"], offset_ms=j["offset_ms"],
                  sched_err_ms=round((r["t_write"] - self.press_at) * 1000, 2),
                  pop_to_press_ms=round((r["t_write"] - j["pop_ts"]) * 1000, 1),
                  pop_est_to_press_ms=round((r["t_write"] - j["pop_est"]) * 1000, 1), center_ms=j["center_ms"], scale=round(j["scale"], 3), first_cue=j["first_cue"], hint_at_press=j.get("hint_at_press"), kernel_lag_ms=r.get("kernel_lag_ms"), interval_ms=round(j["interval_ms"], 1) if j["interval_ms"] else None,
                  pop_gap_ms=round(j["pop_gap_ms"], 1) if j["pop_gap_ms"] else None, pop_score=round(j["pop_score"], 3),
                  write_ms=round((r["t_done"] - r["t_write"]) * 1000, 3), ok=r["ok"], err=r.get("err"))
        self.cur = None
        self.pending.append(j)
        self.hint_at_press = j["hint_at_press"]
        self.hint_until, self.hint_press_ts = r["t_write"] + 0.5, r["t_write"]
        self.state, self.release_at = "HELD", r["t_write"] + self.tune["hold_ms"] / 1000
        if not r["ok"]:
            self.stats["inject_fail"] += 1

    # ---- 结果与归因
    def _block(self, t, j, success):
        b = self.blk
        b["ok" if success else "fail"] += 1; b["off"].append(j["offset_ms"]); b["n"] += 1
        if (b["ok"] + b["fail"]) % 10 == 0:
            med = lambda a: round(float(np.median(a)), 1) if a else None
            self.emit("block10", t, index=b["n"] // 10, ok=b["ok"], fail=b["fail"],
                      interval_ms_median=med(b["ivl"][-10:]), interval_ms_min=round(min(b["ivl"][-10:]), 1) if b["ivl"] else None,
                      interval_ms_max=round(max(b["ivl"][-10:]), 1) if b["ivl"] else None,
                      visible_ms_median=med(b["vis"][-10:]), offset_ms_used=med(b["off"][-10:]),
                      eval_ms_median=med(b.get("eval", [])), margin_ms_min=round(min(b["margin"]), 1) if b.get("margin") else None,
                      margin_ms_max=round(max(b["margin"]), 1) if b.get("margin") else None)
            b["ivl"], b["vis"], b["off"], b["ok"], b["fail"] = [], [], [], 0, 0
            b["eval"], b["margin"] = [], []

    def _resolve(self, j, success, t, game_over=False):
        self._block(t, j, success)
        if success:
            self.stats["success"] += 1
            self.last_success = t
            self.consec = collections.Counter()
            n = self.stats["success"]
            if n in (1, 20, 50, 100, 200, 300, 500, 700, 1000):
                self.on_dump(f"milestone{n}", t)
            if n % 20 == 0 and self.ivl_hist:
                self.emit("tempo", t, successes=n, interval_median_s=round(float(np.median(self.ivl_hist)), 3),
                          interval_max_s=round(max(self.ivl_hist), 3), visible_ms_median=round(float(np.median(self.vis_hist)), 1) if self.vis_hist else None)
            self._bin(j, True)
            self.emit("result", t, jump=j["id"], ok=True, counter_latency_ms=round((t - j["press_ts"]) * 1000, 1),
                      eval_ms=round((t - j["pop_ts"]) * 1000, 1), interval_ms=round(j["interval_ms"], 1) if j["interval_ms"] else None)
            self.blk.setdefault("eval", []).append((t - j["pop_ts"]) * 1000)
            self.blk.setdefault("margin", []).append((t - j["press_ts"]) * 1000)
            return
        cause, why = self._classify(j)
        self.stats["fail_" + cause] += 1
        pt = j["press_ts"]
        why = {**why, "scores_ms_rel_press": [[round((ts_ - pt) * 1000), round(sc, 2)] for ts_, sc in list(self.score_hist)[-22:]],
               "offset_ms": round(j["offset_ms"], 1), "scale": round(j["scale"], 3),
               "interval_ms": round(j["interval_ms"], 1) if j["interval_ms"] else None}
        self.emit("result", t, jump=j["id"], ok=False, cause=cause, evidence=why, game_over=game_over)
        self.on_dump("fail_" + cause, t)
        if cause == "inject_fail":
            return                                  # 与时序无关，不进入时序统计
        self._bin(j, False)
        if cause == "ignored_early":
            self.consec["early"] += 1; self.consec["late"] = 0
        elif cause == "late_after_vanish":
            self.consec["late"] += 1; self.consec["early"] = 0
        else:
            self.consec = collections.Counter()
        self._maybe_adjust(t)

    def _classify(self, j):
        pt = j["press_ts"]
        if not j.get("inj_ok", True):
            return "inject_fail", {"err": "按键写入失败或内核无回显"}
        if "vanish_ts" not in j or j["vanish_ts"] < j["pop_ts"]:
            return "ignored_early", {"note": "按键后气泡始终未消失：游戏未响应，倾向按得过早"}
        if j["vanish_ts"] < pt:
            return "late_after_vanish", {"vanish_before_press_ms": round((pt - j["vanish_ts"]) * 1000, 1)}
        hp = j.get("hint_at_press")
        if hp == "A":
            rec = ({"received": True, "hint_switch_ms_after_press": round(j["hint_ms"], 1),
                    "processed_ms_after_pop": round(j["offset_ms"] + j["hint_ms"], 1)} if "hint_ms" in j
                   else {"received": False, "note": "按键时提示为手柄 A，500ms 内没有切换为键盘 F：游戏很可能没有处理到这次按键"})
        else:
            rec = {"received": None, "note": "按键时提示已是 F(或无法判别)，无法用图标判断游戏是否收到"}
        return "no_score", {"reception": rec, "vanish_after_press_ms": round((j["vanish_ts"] - pt) * 1000, 1),
                            "note": "按键已写入内核且有回显，但没有得分。无法区分：游戏没收到 / 收到但在有效窗口外(过早或过晚)；"
                                    "uinput 写入成功不能证明游戏接收，见 hint_changed 事件(提示图标切换=游戏收到键盘输入)"}

    def _bin(self, j, success):
        b = str(int(round(j["offset_ms"] / 5) * 5))
        s, f = self.tune["bins"].get(b, [0, 0])
        self.tune["bins"][b] = [s + (1 if success else 0), f + (0 if success else 1)]

    def _maybe_adjust(self, t):
        """只在有方向性证据的连续事件上调整：连续 2 次同类归因才动"""
        lo, hi = LIMITS["offset_ms"]
        off = self.tune["offset_ms"]
        if self.consec["early"] >= 2 and off < hi:
            self._set("offset_ms", min(hi, off + 20), t, "连续2次按键后气泡仍未消失(ignored_early)")
            self.consec = collections.Counter()
            if off >= 230:
                self._set("hold_ms", min(LIMITS["hold_ms"][1], self.tune["hold_ms"] + 20), t, "offset 已近上限仍被忽略，延长按住")
        elif self.consec["late"] >= 2 and off > lo:
            self._set("offset_ms", max(lo, off - 20), t, "连续2次气泡在按键前已消失(late)")
            self.consec = collections.Counter()

    def _visual_miss(self, t):
        self.miss_flagged = True
        if self.dark_since_pop:                      # 提示停了且屏幕变暗：挑战通关/失败结算，不是漏检
            self.emit("end_of_run", t, since_last_pop_s=round(t - self.last_pop, 2), max_score_in_gap=round(self.gap_max, 3))
            return
        self.stats["visual_miss"] += 1
        self.emit("visual_miss", t, since_last_pop_s=round(t - self.last_pop, 2), max_score_in_gap=round(self.gap_max, 3))
        self.on_dump("visual_miss", t)
        if 0.5 <= self.gap_max < self.tune["thr_on"]:  # 有迹象但没过阈值：阈值偏高
            self._set("thr_on", max(LIMITS["thr_on"][0], round(self.tune["thr_on"] - 0.03, 3)), t,
                      f"漏检窗口内最高得分 {self.gap_max:.2f} 低于阈值")

    def _set(self, key, val, t, why):
        old = self.tune[key]
        if val == old:
            return
        self.tune[key] = val
        self.emit("param_change", t, param=key, old=old, new=val, evidence=why)

    # ---- 跨局更新（有证据才动）
    def finish_run(self, t):
        bins = {int(k): v for k, v in self.tune["bins"].items()}
        cur = int(round(self.tune["offset_ms"] / 5) * 5)

        def lb(b):
            s, f = bins.get(b, [0, 0])
            return wilson_lb(s, s + f)
        cand = [b for b, (s, f) in bins.items() if s + f >= 8]
        if sum(bins.get(cur, [0, 0])) < 8:           # 当前中心延迟样本不足：继续收集，不拿"没数据"当"成功率 0"去比较
            return
        if cand:
            best = max(cand, key=lb)
            if best != cur and lb(best) > lb(cur) + 0.1:
                step = max(-20, min(20, best - cur))
                self._set("offset_ms", float(cur + step), t, f"跨局统计：{best}ms 成功率下界 {lb(best):.2f} > 当前 {lb(cur):.2f}")
                return
        s, f = bins.get(cur, [0, 0])
        if self.stats["fail_no_score"] and f >= 2 and f > s:
            lo_n = sum(bins.get(cur - 20, [0, 0])); hi_n = sum(bins.get(cur + 20, [0, 0]))
            step = 20 if hi_n <= lo_n else -20     # 向样本更少的一侧探索
            new = max(LIMITS["offset_ms"][0], min(LIMITS["offset_ms"][1], cur + step))
            self._set("offset_ms", float(new), t, f"当前 {cur}ms 失败 {f} 成功 {s}，无方向性证据，向样本更少一侧探索")


# ================================================================ 运行
class Runner:
    def __init__(self, args):
        self.args = args
        self.cfg = load_json(CFG_PATH, DEFAULTS)
        if args.first_offset_ms:
            self.cfg["first_cue_offset_ms"] = args.first_offset_ms
        self.tune = load_tune(args.baseline)
        self.cfg_overrides = bool(args.offset_ms or args.hold_ms or args.first_offset_ms)
        if args.offset_ms: self.tune["offset_ms"] = args.offset_ms
        if args.hold_ms: self.tune["hold_ms"] = args.hold_ms
        self.stop = threading.Event()
        self.focused = True
        self.tag = time.strftime("run-%Y%m%d-%H%M%S")
        (HERE / "logs").mkdir(exist_ok=True)
        if args.dump_frames:
            (HERE / "logs" / "dumps").mkdir(exist_ok=True)
        self.log = logging.getLogger("ffjump"); self.log.setLevel(logging.DEBUG)
        fh = logging.FileHandler(HERE / "logs" / f"{self.tag}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%H:%M:%S"))
        fh.setLevel(logging.INFO)
        sh = logging.StreamHandler(); sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        self.log.addHandler(fh); self.log.addHandler(sh)
        self.events = open(HERE / "logs" / f"{self.tag}.jsonl", "w", buffering=1)
        self.t0 = now(); self.wall0 = time.time() - self.t0
        self.ring = collections.deque(maxlen=150) if args.dump_frames else None   # 仅诊断模式才在内存保留帧
        self.dumpq = queue.Queue()
        self.key = self.engine = self.cap = None
        self.frames = self.dups = self.skipped = 0
        self.intervals, self.proc_ms = [], []
        self.toggle_req = threading.Event()
        self.trace = None
        if args.trace:                                         # 仅诊断模式：逐帧写盘
            self.trace = open(HERE / "logs" / f"{self.tag}.frames.csv", "w", buffering=1 << 16)
            self.trace.write("t,gap_ms,lag_ms,proc_ms,score\n")

    # ---- 事件/日志
    def emit(self, typ, t, **kw):
        ev = {"type": typ, "t": round(t - self.t0, 4), "wall": round(t + self.wall0, 3), **kw}
        self.events.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        msg = {"press": lambda: f"F 按下 #{kw['jump']} pop+{kw['pop_to_press_ms']}ms 注入写入{kw['write_ms']}ms ok={kw['ok']}",
               "result": lambda: (f"#{kw['jump']} 成功 (计数 +{kw['counter_latency_ms']}ms)" if kw["ok"]
                                  else f"#{kw['jump']} 失败 原因={kw['cause']} {kw['evidence']}"),
               "pop": lambda: f"「！」弹出 score={kw['score']}", "dark_screen": lambda: f"屏幕变暗(仅记录，不影响开关) 亮度{kw['brightness']}/基线{kw['baseline']}",
               "enabled": lambda: f"== 自动跳绳 已开启 {self.engine.params()} ==",
               "disabled": lambda: f"== 自动跳绳 已关闭({kw['reason']}) ==",
               "param_change": lambda: f"参数调整 {kw['param']}: {kw['old']} -> {kw['new']}  依据: {kw['evidence']}",
               "end_of_run": lambda: f"提示停止且屏幕变暗：通关或失败结算(不算漏检) {kw}", "tempo": lambda: f"节拍 {kw}", "cancel": lambda: f"取消触发 {kw['why']}", "visual_miss": lambda: f"漏检 {kw}"}
        if typ in msg:
            self.log.info(msg[typ]())
        # 其余事件只写 jsonl，不再重复写文本日志

    def dump(self, reason, t):
        if self.ring is None:
            return
        frames = list(self.ring)
        if frames and len(list((HERE / "logs" / "dumps").glob("*"))) < 300:
            self.dumpq.put((HERE / "logs" / "dumps" / f"{self.tag}-{reason}-{t - self.t0:.2f}.npz", frames, t))

    def _dump_worker(self):
        while True:
            p, frames, t = self.dumpq.get()
            try:
                np.savez_compressed(p, ts=np.array([f[0] for f in frames]), gray=np.stack([f[1] for f in frames]), t_event=t)
            except Exception as e:
                self.log.warning("写片段失败: %s", e)
            self.dumpq.task_done()

    # ---- 控制
    def _toggle(self):
        self.toggle_req.set()
        if self.cap:
            self.cap.wake()

    def _apply_toggle(self, t):
        if self.engine.enabled:
            self.engine.disable(t, "user")
            self.engine.finish_run(t); self.save_tune()
        else:
            self.engine.enable(t)

    def save_tune(self):
        if self.cfg_overrides:                      # 命令行临时实验值不写入学习结果
            return
        TUNE_PATH.write_text(json.dumps(self.tune, indent=1, ensure_ascii=False) + "\n")

    def _on_signal(self, signum, _):
        if self.key and self.key.down:
            try:
                self.key.release()
            except Exception:
                pass
        self.stop.set()
        if self.cap:
            self.cap.wake()

    def _focus_loop(self):
        while not self.stop.is_set():
            try:
                self.focused = sh_json("hyprctl", "activewindow", "-j").get("class") == self.cfg["window_class"]
            except Exception:
                pass
            time.sleep(0.3)

    def summary(self):
        el = now() - self.t0
        p = lambda a, q: round(float(np.percentile(a, q)), 3) if a else None
        st = self.engine.stats
        press = st["presses"]
        return {"runtime_s": round(el, 1), "frames": self.frames, "fps": round(self.frames / el, 1),
                "unique_frames": self.frames - self.dups, "loop_skipped": self.skipped,
                "interval_ms_p50_p99_max": [p(self.intervals, 50), p(self.intervals, 99), round(max(self.intervals), 1) if self.intervals else None],
                "process_ms_p50_p99": [p(self.proc_ms, 50), p(self.proc_ms, 99)],
                "pops": st["pops"], "presses": press, "success": st["success"],
                "success_rate": round(st["success"] / press, 3) if press else None,
                "failures": {k[5:]: v for k, v in st.items() if k.startswith("fail_")},
                "cancels": {"cue_gone": st["cancel_gone"], "repeat_blocked": st["repeat_blocked"]},
                "visual_miss": st["visual_miss"], "inject_fail": st["inject_fail"], "params": self.engine.params()}

    # ---- 主流程
    def run(self):
        a, cfg, log = self.args, self.cfg, self.log
        if a.dump_frames:
            threading.Thread(target=self._dump_worker, daemon=True).start()
        tpl = cv2.imread(str(HERE / cfg["template"]), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            sys.exit(f"模板不存在: {cfg['template']}")
        self.key = FakeKey() if a.dry_run else UInputKey(log)
        self.engine = Engine(cfg, self.tune, tpl, self.key, self.emit, focused=lambda: self.focused)
        if a.dump_frames:
            self.engine.on_dump = self.dump
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, self._on_signal)
        if cfg["focus_check"] and not a.no_focus_check:
            threading.Thread(target=self._focus_loop, daemon=True).start()
        ToggleKey(self._toggle, log).start()
        log.info("待命，不会按 F。按一次 NumPad 0 开启，再按一次关闭；Ctrl+C 退出。%s", "(dry-run: 不发键)" if a.dry_run else "")
        log.info("参数: %s", json.dumps(self.engine.params()))
        if a.arm_at_start and a.dry_run:
            self.engine.enable(now())
        try:
            while not self.stop.is_set():
                try:
                    geom = Geometry(cfg)
                    self._session(geom, tpl)
                except Exception as e:                 # 窗口暂时找不到/捕获失败：重试，绝不改变开关
                    log.warning("捕获会话异常(%s)，1 秒后重试；开关状态保持 %s", e, "开启" if self.engine.enabled else "待命")
                    time.sleep(1.0)
        finally:
            t = now()
            if self.engine.enabled:
                self.engine.disable(t, "exit")
            if self.trace:
                self.trace.close()
            self.engine.finish_run(t); self.save_tune()
            if self.key:
                self.key.close()
            self.dumpq.join()
            s = self.summary()
            log.info("统计: %s", json.dumps(s, ensure_ascii=False, indent=1))
            json.dump(s, open(HERE / "logs" / f"{self.tag}.summary.json", "w"), indent=1, ensure_ascii=False)

    def _session(self, geom, tpl):
        cfg, log, eng = self.cfg, self.log, self.engine
        cap = self.cap = Capture(geom.region(), log); cap.start()
        by0, by1, bx0, bx1 = geom.sub(cfg, cfg["bubble_ref"])
        cy0, cy1, cx0, cx1 = geom.sub(cfg, cfg["counter_ref"])
        hy0, hy1, hx0, hx1 = geom.sub(cfg, cfg["hint_ref"])
        W, H = cap.size
        if by1 > H + 4 or cy1 > H + 4 or bx1 > W + 4:
            log.error("捕获帧尺寸 %s 与子区域不符", cap.size)
        by1, cy1, bx1, cx1 = min(by1, H), min(cy1, H), min(bx1, W), min(cx1, W)   # 取整误差 1-2px
        k = geom.k / cfg["template_k"]
        eng.tpl = tpl if abs(k - 1) <= 0.02 else cv2.resize(tpl, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        log.info("窗口 %s,%s %sx%s k=%.3f 捕获 %s -> %sx%s 气泡区 y%d:%d 计数区 y%d:%d x%d:%d",
                 geom.win_x, geom.win_y, geom.win_w, geom.win_h, geom.k, geom.region(), *cap.size, by0, by1, cy0, cy1, cx0, cx1)
        last_idx, prev, last_ts, last_geo = 0, None, None, now()
        last_unique_ts = None
        try:
            while not self.stop.is_set():
                dl = eng.next_deadline()
                timeout = 0.2 if dl is None else max(0.0, min(0.2, dl - now()))
                idx, ts, buf, dead = cap.wait(last_idx, timeout)
                t = now()
                if dead:
                    log.error("捕获进程退出，1 秒后重启捕获（自动跳绳开关状态保持不变）")
                    time.sleep(1.0); return
                if self.toggle_req.is_set():
                    self.toggle_req.clear(); self._apply_toggle(t)
                if idx == last_idx:
                    eng.tick(t)                     # 无新帧时仍要处理到期的按键/释放/判定
                    if t - last_geo > 1.0:
                        last_geo = t
                        try:
                            if Geometry(cfg).key != geom.key:
                                log.warning("窗口位置/大小/缩放变化，重新初始化"); return
                        except Exception as e:
                            log.warning("几何检查失败: %s", e)
                    continue
                if last_idx:
                    self.skipped += idx - last_idx - 1
                last_idx = idx
                self.frames += 1
                if last_ts is not None:
                    self.intervals.append((ts - last_ts) * 1000)
                last_ts = ts
                gray = frame_to_gray(buf, cap.size)
                if prev is not None and np.array_equal(gray, prev):
                    self.dups += 1
                    eng.tick(t)
                    continue                        # 重复帧：画面没更新，无需重算
                prev = gray
                if self.ring is not None:
                    self.ring.append((ts, gray))
                bright = float(gray[::8, ::8].mean())
                t0 = now()
                sc = eng.on_frame(ts, gray[by0:by1, bx0:bx1], gray[cy0:cy1, cx0:cx1], bright, gray[hy0:hy1, hx0:hx1])
                t1 = now()
                self.proc_ms.append((t1 - t0) * 1000)
                if self.trace:
                  self.trace.write(f"{ts - self.t0:.4f},{(ts - last_unique_ts) * 1000 if last_unique_ts else 0:.1f},{(t0 - ts) * 1000:.2f},{(t1 - t0) * 1000:.2f},{sc:.3f}\n")
                last_unique_ts = ts
        finally:
            cap.stop()
            self.cap = None


# ================================================================ 离线回放
def cmd_replay(args):
    cfg = load_json(CFG_PATH, DEFAULTS)
    tune = {**TUNE_DEFAULTS, "bins": {}}
    tpl = cv2.imread(str(HERE / cfg["template"]), cv2.IMREAD_GRAYSCALE)
    total = 0
    for path in args.clips:
        z = np.load(path)
        ts, g, te = z["ts"], z["gray"], float(z["t_event"])
        events = []
        vt = [0.0]
        full = g.shape[1] > 400 and g.shape[2] > 300 and g.shape[1] >= 580      # run-*.npz：完整 ROI 帧
        geo = type("G", (), {"k": 1.25})()
        by0, by1, bx0, bx1 = Geometry.sub(geo, cfg, cfg["bubble_ref"]); cy0, cy1, cx0, cx1 = Geometry.sub(geo, cfg, cfg["counter_ref"])
        hy0, hy1, hx0, hx1 = Geometry.sub(geo, cfg, cfg["hint_ref"])
        eng = Engine(cfg, tune, tpl, FakeKey(lambda: vt[0]), lambda typ, t, **kw: events.append((typ, t, kw)), have_counter=full)
        eng.enable(ts[0])
        for i in range(len(g)):
            if i:                                   # 帧间以 1ms 步进模拟定时器
                t = ts[i - 1]
                while t < ts[i]:
                    vt[0] = t; eng.tick(t); t += 0.001
            vt[0] = ts[i]
            if full:
                eng.on_frame(ts[i], g[i][by0:by1, bx0:bx1], g[i][cy0:cy1, cx0:cx1], float(g[i][::8, ::8].mean()), g[i][hy0:hy1, hx0:hx1])
            else:
                eng.on_frame(ts[i], g[i])
        f = lambda n: [round(e[1] - te, 3) for e in events if e[0] == n]
        total += len(f("press"))
        if full:
            for typ, t, kw in events:
                if typ in ("enabled", "disabled", "pop", "press", "vanish", "result", "dark_screen", "counter_change", "cancel", "hint_changed"):
                    print(f"   {t - ts[-1]:+7.3f}s {typ} { {k: v for k, v in kw.items() if k != 'loc'} }")
            print(f"   结束时 enabled={eng.enabled}")
        print(f"{Path(path).name:28s} 人工F=0  pop={[float(x) for x in f('pop')]} vanish={[float(x) for x in f('vanish')]}  程序按键={[float(x) for x in f('press')]}")
    print("程序按键总数:", total)


def cmd_info(_):
    cfg = load_json(CFG_PATH, DEFAULTS)
    print("session:", os.environ.get("XDG_SESSION_TYPE"), os.environ.get("XDG_CURRENT_DESKTOP"))
    g = Geometry(cfg)
    print(f"game window at {g.win_x},{g.win_y} {g.win_w}x{g.win_h} k={g.k:.3f} capture {g.region()}")
    print("wf-recorder", subprocess.run(["which", "wf-recorder"], capture_output=True, text=True).stdout.strip() or "MISSING")
    print("/dev/uinput writable:", os.access("/dev/uinput", os.W_OK), " cv2", cv2.__version__)
    print("tuning:", json.dumps({k: v for k, v in load_json(TUNE_PATH, TUNE_DEFAULTS).items() if k != "bins"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true", help="只检测记录，不发 F")
    r.add_argument("--arm-at-start", action="store_true", help="(仅 dry-run 测试用) 启动即开启")
    r.add_argument("--no-focus-check", action="store_true")
    r.add_argument("--offset-ms", type=float, help="(实验) 本次运行临时延迟，不写入 tuning.json")
    r.add_argument("--hold-ms", type=float, help="(实验) 本次运行临时按住时长，不写入 tuning.json")
    r.add_argument("--first-offset-ms", type=float, help="(实验) 仅新局第一个提示使用的延迟")
    r.add_argument("--baseline", action="store_true", help="忽略 tuning.json，只用 baseline.json 的稳定基线")
    r.add_argument("--dump-frames", action="store_true", help="(诊断) 在失败等事件保存前后帧；默认关闭")
    r.add_argument("--trace", action="store_true", help="(诊断) 逐帧写 csv；默认关闭")
    p = sub.add_parser("replay"); p.add_argument("clips", nargs="+")
    a = ap.parse_args()
    {"info": cmd_info, "replay": cmd_replay}.get(a.cmd, lambda a: Runner(a).run())(a)


if __name__ == "__main__":
    main()
