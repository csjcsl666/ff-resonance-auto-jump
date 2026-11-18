# ff-resonance-auto-jump

Final Fantasy Resonance 跳绳小游戏自动跳跃工具

检测角色头顶出现的「！」提示，自动按 F，用小键盘 NumPad 0 开启和关闭

## 适用环境

- 游戏：Final Fantasy Resonance（Steam，Proton 运行，窗口类名 steam_app_3259780）
- 系统：Linux，Wayland，Hyprland（用 hyprctl 查询游戏窗口）
- 已实测的环境：CachyOS，Hyprland 0.56.2，wf-recorder 0.6.0，Python 3.12，OpenCV 5.1.0，显示器 2560x1440 缩放 1.25，游戏窗口 2048x1152 全屏
- 其他合成器、分辨率和语言版本没有测试

## 功能

- 持续检测跳绳小游戏里的「！」提示，出现后按下 F，按住 150 毫秒
- 延迟基线是 110 毫秒，最近几个提示的可见时间较短时会按比例缩短，实测运行中为 77 到 110 毫秒，中位 85 毫秒
- 启动后默认待命，不会按键，按一次 NumPad 0 开启，再按一次关闭，关闭时立即松开 F
- 只在游戏窗口是焦点时发送按键
- 用计数数字的变化判断每次跳跃是否成功，失败原因写入日志，连续出现方向一致的失败时小幅调整延迟
- 游戏画面变暗（转场）只写日志，不会关闭自动跳绳
- 默认不录屏，不保存画面，只在内存里检测

## 实测结果

- 60 次挑战每 10 次变速一次，一次程序运行里累计 206 次跳跃全部成功（中途手动关闭过一次）
- 参数：延迟基线 110 毫秒（实际 77 到 110 毫秒），按住 150 毫秒，识别阈值 0.70

## 安装

```
git clone https://github.com/csjcsl666/ff-resonance-auto-jump
cd ff-resonance-auto-jump
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

系统工具：

- wf-recorder（Arch 系：sudo pacman -S wf-recorder）

权限：

- 监听 NumPad 0 需要读取 /dev/input，发送 F 需要写 /dev/uinput
- 普通用户通常需要加入 input 组并重新登录：sudo usermod -aG input $USER
- 用 ls -l /dev/uinput 检查当前用户是否有写权限
- 本程序不会替你修改任何系统设置

## 使用

```
.venv/bin/python ffjump.py run
```

- 第 1 步：终端出现「待命」说明启动成功，此时不会按键
- 第 2 步：进入跳绳小游戏
- 第 3 步：按一次 NumPad 0 开启自动跳绳
- 第 4 步：再按一次 NumPad 0 关闭
- 第 5 步：按 Ctrl+C 退出程序，退出时会松开 F 并销毁虚拟键盘

说明：

- NumPad 0 也会被游戏收到，它在游戏里没有功能
- 只想查看识别而不发送按键，加上 --dry-run
- 不需要手动跳绳采样，也没有其他快捷键

## 配置

- baseline.json：已验证的稳定参数（延迟基线 110，按住 150，识别阈值 0.70），程序不会写这个文件
- ffjump.json：可选，自行创建，用来覆盖默认配置，例如 {"window_class": "steam_app_数字"}
- --offset-ms 和 --hold-ms：本次运行临时修改延迟和按住时长，不写入学习结果
- --baseline：忽略 tuning.json，只使用 baseline.json
- tuning.json：运行中自动学习的结果，已在 .gitignore 里，不会提交

## 日志

运行时在 logs/ 目录生成：

- run-时间.log：文本日志
- run-时间.jsonl：事件记录（提示弹出、按键、结果、每 10 次统计）
- run-时间.summary.json：本次运行统计

默认不保存任何画面

## 测试

```
.venv/bin/python tests/test_engine.py
```

测试使用合成画面和虚拟时钟，不会向游戏发送按键

## 已知问题

- 持续抓取游戏画面会让 Hyprland 的 CPU 占用明显升高，桌面可能变卡，不使用时请退出程序
- 游戏窗口移动或改变大小时会自动重新初始化捕获
- 识别依赖游戏界面里的「！」图标和计数数字，游戏更新或更换分辨率后，templates 里的模板可能需要更新
- 只测试了单机跳绳小游戏，没有测试其他玩法
- 长时间使用后（尤其是长时间持续抓屏之后）Hyprland 可能整体变慢，表现为桌面卡顿、自动跳绳连续失败
- 实测重启系统后恢复，退出游戏后桌面仍然卡顿，只重启游戏或重新登录 Hyprland 会话能否恢复都没有验证，具体根因尚未确定

## 声明

- 不读取或修改游戏文件和内存，只读取屏幕画面并模拟键盘 F 键
- 仅用于个人单机游戏，使用风险自负
- 非官方项目，与游戏发行方无关，游戏名称和界面截图的版权归其所有者

## 许可证

MIT
