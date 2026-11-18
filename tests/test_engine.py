"""Engine 逻辑测试：合成帧 + 虚拟时钟 + 假按键，不碰真实键盘/游戏"""
import sys, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, cv2
import ffjump as F

HERE = F.HERE
tpl = cv2.imread(str(HERE / "templates/bubble_A.png"), 0)
rng = np.random.default_rng(1)
BG = rng.integers(60, 140, (300, 300)).astype(np.uint8)
GL = [((rng.random((48, 112)) < 0.12) * 255).astype(np.uint8) for _ in range(400)]   # 互不相同的"数字"

def bub(present, score_scale=1.0):
    f = BG.copy()
    if present:
        t = tpl.astype(np.float32)
        t = (BG[100:100 + t.shape[0], 100:100 + t.shape[1]] * (1 - score_scale) + t * score_scale).astype(np.uint8)
        f[100:100 + tpl.shape[0], 100:100 + tpl.shape[1]] = t
    return f

class Rig:
    def __init__(self, **tune):
        self.cfg = F.load_cfg(); self.cfg["dither_ms"] = tune.pop("dither", [0]); self.cfg["scale_with_visible"] = tune.pop("scale", False)
        self.tune = {**F.TUNE_DEFAULTS, "bins": {}, "offset_ms": 110.0, **tune}
        self.t = 100.0; self.ev = []; self.focus = True
        self.key = F.FakeKey(lambda: self.t); self.key.down = False
        orig = self.key.press
        self.fail = False
        def press():
            r = orig(); self.key.down = True
            if self.fail: r["ok"] = False; r["err"] = "x"
            return r
        def release():
            self.key.down = False
        self.key.press, self.key.release = press, release
        self.e = F.Engine(self.cfg, self.tune, tpl, self.key, lambda typ, t, **kw: self.ev.append((typ, t, kw)), focused=lambda: self.focus)
        self.e.enable(self.t)
        self.count = 0; self.bright = 90.0
    def step(self, present, dt=0.02, count_changed=False, scale=1.0):
        self.t += dt
        if count_changed: self.count += 1
        self.e.on_frame(self.t, bub(present, scale), GL[self.count], self.bright)
    def run(self, secs, present, **kw):
        end = self.t + secs
        while self.t < end: self.step(present, **kw)
    def n(self, typ): return [e for e in self.ev if e[0] == typ]

def jump(r, ok=True, persist=False):
    """一次完整的提示-按键-结果（节拍约 0.88s）；返回 pop 时刻"""
    r.run(0.2, False)
    pop = r.t + 0.02
    r.run(0.13, True)              # 弹出后到触发
    r.run(0.10, True)              # 触发后，气泡约 100ms 后消失
    if persist:
        r.run(0.9, True)           # 气泡不消失
    else:
        r.run(0.10, False)
        if ok: r.step(False, count_changed=True)
        r.run(0.3, False)
    return pop

def test_success_chain():
    r = Rig()
    for _ in range(10): jump(r)
    s = r.e.stats
    assert s["presses"] == 10 and s["success"] == 10, dict(s)
    pr = r.n("press")
    assert all(95 <= p[2]["pop_to_press_ms"] <= 135 for p in pr), [p[2]["pop_to_press_ms"] for p in pr]
    assert not r.n("param_change") and r.e.consec["early"] == 0
    print("success_chain ok: 10/10, pop->press", sorted({round(p[2]['pop_to_press_ms']) for p in pr}))

def test_no_repeat_on_persistent_bubble():
    r = Rig(); r.run(0.4, False); r.run(3.0, True)
    assert r.e.stats["presses"] == 1, r.e.stats["presses"]
    print("persistent bubble -> 1 press ok")

def test_ignored_early_adjusts_after_two():
    r = Rig()
    jump(r, ok=True)
    jump(r, ok=False, persist=True); assert not r.n("param_change")      # 1 次不动
    jump(r, ok=False, persist=True)
    pc = r.n("param_change")
    assert len(pc) == 1 and pc[0][2]["param"] == "offset_ms" and pc[0][2]["new"] == 130.0, pc
    fails = [e[2]["cause"] for e in r.n("result") if not e[2]["ok"]]
    assert fails == ["ignored_early"] * 2, fails
    print("ignored_early x2 -> offset 110->130 ok")

def test_ambiguous_does_not_adjust():
    r = Rig()
    for _ in range(3):
        r.run(0.2, False); r.run(0.13, True); r.run(0.10, True); r.run(0.1, False); r.run(0.9, False)  # 气泡消失但计数没变
    fails = [e[2]["cause"] for e in r.n("result") if not e[2]["ok"]]
    assert fails == ["no_score"] * 3, fails
    assert not r.n("param_change")
    print("ambiguous x3 -> 不改参数(无方向证据) ok")

def test_inject_fail_not_timing():
    r = Rig(); r.fail = True
    for _ in range(3): jump(r, ok=False)
    fails = [e[2]["cause"] for e in r.n("result") if not e[2]["ok"]]
    assert fails and set(fails) == {"inject_fail"} and not r.tune["bins"] and not r.n("param_change"), (fails, r.tune["bins"])
    print("inject_fail 归因且不进入时序统计 ok")

def test_not_focused():
    r = Rig(); r.focus = False; jump(r)
    assert r.e.stats["presses"] == 0 and r.n("cancel")[0][2]["why"] == "not_focused"
    print("not_focused -> 不发 F ok")

def test_dark_screen_never_disables():
    """开场/失败后的淡入淡出：只记录 dark_screen，开关保持开启，后续照常按键，且不污染参数"""
    r = Rig()
    jump(r); jump(r)
    r.bright = 5.0
    r.run(0.3, False)
    assert r.n("dark_screen") and r.e.enabled and r.e.state == "ARMED" and not r.n("disabled")
    r.bright = 90.0; r.run(0.5, False)
    n = r.e.stats["presses"]
    for _ in range(3): jump(r)
    assert r.e.stats["presses"] == n + 3 and r.e.enabled
    assert not r.n("param_change") and not [e for e in r.n("result") if not e[2]["ok"]]
    print("暗屏只记录、开关保持开启、之后继续按键 ok")

def test_fade_does_not_fake_counter_success():
    """淡入淡出造成的计数区变化不能算成功"""
    r = Rig()
    r.run(1.0, False)                                # 先积累亮度基线
    r.run(0.2, False); r.run(0.15, True)            # 触发按键
    assert r.e.stats["presses"] == 1
    r.bright = 20.0                                  # 淡出，计数区消失 + 变化
    r.count += 7; r.run(0.1, False)
    r.bright = 90.0
    r.run(0.8, False)
    res = r.n("result")
    assert res and not res[0][2]["ok"], res          # 不是成功
    print("淡入淡出不会被当成计数成功 ok")

def test_disable_releases_immediately():
    r = Rig(); r.run(0.4, False); r.run(0.16, True)
    assert r.key.down, "应已按下"
    r.e.disable(r.t, "user")
    assert not r.key.down and r.e.state == "OFF"
    r.run(1.0, True); assert r.e.stats["presses"] == 1
    print("关闭 -> 立即松开 F 且不再触发 ok")

def test_visual_miss_lowers_threshold():
    r = Rig()
    jump(r); jump(r)
    r.run(0.4, False)
    r.run(0.5, True, scale=0.28)     # 弱提示：得分高于 thr_off 但低于 thr_on
    r.run(2.5, False)
    vm = r.n("visual_miss")
    print("  max_score_in_gap:", vm and vm[0][2]["max_score_in_gap"])
    assert vm and 0.5 <= vm[0][2]["max_score_in_gap"] < 0.70, vm
    pc = r.n("param_change")
    assert pc and pc[0][2]["param"] == "thr_on" and pc[0][2]["new"] == 0.67, pc
    print("visual_miss(弱提示 0.5-0.7) -> thr_on 0.70->0.67 ok")

def test_dither_varies_within_range():
    r = Rig(dither=[-15, 0, 15])
    for _ in range(30): jump(r)
    offs = sorted({round(e[2]["offset_ms"]) for e in r.n("press")})
    assert set(offs) == {95, 110, 125}, offs
    assert r.e.stats["success"] == 30
    print("抖动延迟 95/110/125 均出现且全部成功 ok")

def test_adaptive_period_fast_tempo():
    """游戏若分阶段加速(节拍 0.5s)：重复保护与漏检判定须跟随，不能误拦"""
    r = Rig()
    def fast():
        r.run(0.05, False); r.run(0.13, True); r.run(0.07, True); r.run(0.05, False); r.step(False, count_changed=True); r.run(0.13, False)
    for _ in range(14): fast()
    assert r.e.period < 0.6, r.e.period
    assert r.e.stats["repeat_blocked"] == 0, r.e.stats
    print("节拍自适应 period=%.2f 无误拦 ok (按键%d)" % (r.e.period, r.e.stats["presses"]))

def test_end_of_run_not_visual_miss():
    r = Rig()
    for _ in range(3): jump(r)
    r.bright = 5.0; r.run(0.3, False)               # 通关淡出
    r.bright = 90.0; r.run(2.5, False)
    assert not r.n("visual_miss") and not r.n("param_change")
    print("通关淡出后提示停止 -> 不算漏检、不动参数 ok")

def test_baseline_values_and_no_dither_by_default():
    cfg = F.load_cfg(); t = F.load_tune(use_baseline=True)
    assert cfg["dither_ms"] == [] and cfg["pop_midpoint"] is False
    assert (t["offset_ms"], t["hold_ms"], t["thr_on"]) == (110.0, 150.0, 0.70), t
    print("默认：基线 110/150/0.70，无抖动，以检测帧为基准 ok")

def tempo_jump(r, period, vis):
    """period: 提示间隔; vis: 气泡可见时长"""
    r.run(0.1, False)
    r.run(vis * 0.55, True); r.run(vis * 0.45, True)
    r.run(0.05, False); r.step(False, count_changed=True)
    rest = period - 0.1 - vis - 0.05 - 0.02
    if rest > 0: r.run(rest, False)

def test_60_tempo_sim():
    """60 次挑战：每 10 次变速，气泡可见时长按比例缩短；检查全程触发、不误拦、按块统计"""
    r = Rig(scale=True)
    periods = [0.90, 0.80, 0.70, 0.60, 0.52, 0.46]
    for ph, per in enumerate(periods):
        for _ in range(10): tempo_jump(r, per, 0.283 * per)
    s = r.e.stats
    assert s["pops"] == 60 and s["presses"] == 60 and s["repeat_blocked"] == 0 and s["cancel_gone"] == 0, dict(s)
    blocks = r.n("block10")
    assert len(blocks) == 6, len(blocks)
    meds = [round(b[2]["interval_ms_median"]) for b in blocks]
    print("  每10次实测间隔中位 ms:", meds, "按键延迟 ms:", [round(b[2]["offset_ms_used"]) for b in blocks])
    assert meds[0] > meds[-1] and all(abs(m - p * 1000) < 100 for m, p in zip(meds, periods)) and meds == sorted(meds, reverse=True), meds
    assert blocks[-1][2]["offset_ms_used"] < blocks[0][2]["offset_ms_used"]          # 变速后延迟被按比例缩短
    print("60 次变速模拟：60/60 触发，无误拦/取消，6 个块统计 ok")

def test_scale_never_exceeds_baseline():
    r = Rig(scale=True)
    for _ in range(6): jump(r)                       # 可见约 230-250ms
    offs = [e[2]["offset_ms"] for e in r.n("press")]
    assert max(offs) <= 110.0 + 1e-6 and min(offs) >= 44.0, offs
    print("延迟只会缩短不会超过基线 110 ok")

def test_no_dumps_by_default():
    import inspect
    src = inspect.getsource(F.Runner.__init__)
    assert "args.dump_frames" in src and "args.trace" in src
    assert F.Engine(F.load_cfg(), F.load_tune(True), tpl, F.FakeKey(), lambda *a, **k: None).on_dump("x", 0) is None
    print("默认不保存帧/不逐帧写盘 ok")

def test_finish_run_needs_data_at_current():
    r = Rig(offset_ms=90.0)
    r.tune["bins"] = {"110": [41, 2]}               # 只有旧延迟有数据
    r.e.finish_run(r.t)
    assert r.tune["offset_ms"] == 90.0 and not r.n("param_change")
    r.tune["bins"] = {"110": [41, 2], "90": [10, 0]}; r.tune["bins"]["90"] = [3, 6]   # 当前延迟有足够样本且明显更差
    r.e.finish_run(r.t)
    assert r.tune["offset_ms"] == 110.0, r.tune["offset_ms"]
    print("跨局更新：当前延迟无数据不动；有足够数据且明显更差才移动 ok")

def test_uinput_event_bytes():
    """KEY_F 按下/抬起各带一个 EV_SYN；不接触真实设备，写入管道后解码"""
    import os, struct
    rd, wr = os.pipe()
    k = F.UInputKey.__new__(F.UInputKey); k.fd = wr; k.down = False; k.echo_fd = None; k.last_release_lag_ms = None; k.log = None
    k._emit(1); k._emit(0)
    data = os.read(rd, 4096); assert len(data) == 96, len(data)
    ev = [struct.unpack_from("<qqHHi", data, i * 24) for i in range(4)]
    assert [(e[2], e[3], e[4]) for e in ev] == [(1, 33, 1), (0, 0, 0), (1, 33, 0), (0, 0, 0)], ev
    print("uinput 事件字节：KEY_F(33) 按下+SYN，KEY_F 抬起+SYN ok")

def test_hint_classification_and_reception_evidence():
    r = Rig()
    A = r.e.hint_tpl["A"].astype(np.uint8); Fm = r.e.hint_tpl["F"].astype(np.uint8)
    assert r.e._hint_label(A) == "A" and r.e._hint_label(Fm) == "F" and r.e._hint_label(np.zeros_like(A)) is None
    r.run(0.3, False)
    def frame(present, hint):
        r.t += 0.02; r.e.on_frame(r.t, bub(present), GL[r.count], r.bright, hint)
    for _ in range(3): frame(False, A)
    while not r.n("press"): frame(True, A)           # 弹出 -> 按键，图标仍是 A
    assert r.n("press")[0][2]["hint_at_press"] == "A"
    for _ in range(4): frame(True, A)
    for _ in range(3): frame(True, Fm)               # 游戏处理键盘输入后图标变为 F
    hc = r.n("hint_changed"); assert hc and 0 < hc[0][2]["ms_after_press"] < 300, hc
    print("提示图标 A->F 记为输入接收证据，press 事件记录按键时图标 ok")

def test_first_cue_offset_only_when_requested():
    r = Rig(); r.cfg["first_cue_offset_ms"] = 60.0
    for _ in range(3): jump(r)
    ps = r.n("press"); a, b = ps[0][2]["pop_to_press_ms"], ps[1][2]["pop_to_press_ms"]
    assert ps[0][2]["first_cue"] and not ps[1][2]["first_cue"] and a < 85 and b >= 105, (a, b)
    r2 = Rig(); jump(r2)
    assert r2.n("press")[0][2]["pop_to_press_ms"] >= 105          # 默认关闭
    print("首个提示延迟仅在显式指定时生效，默认不变 ok")

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f()
    print("ALL PASSED")
