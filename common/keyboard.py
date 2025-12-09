import threading
from enum import Enum, auto

try:
    from pynput import keyboard as pynput_keyboard
    USE_PYNPUT = True
except ImportError:
    USE_PYNPUT = False
    print("Warning: pynput not installed. Run: pip install pynput")

class KeyboardKey(Enum):
    """键盘按键映射 - 使用不与MuJoCo冲突的按键"""
    # 移动控制 - 使用方向键
    UP = auto()       # 前进
    DOWN = auto()     # 后退
    LEFT = auto()     # 左转
    RIGHT = auto()    # 右转
    
    # 功能键 - 使用数字键 2-9 (MuJoCo只用0和1)
    KEY_2 = auto()    # PASSIVE (阻尼保护)
    KEY_3 = auto()    # POS_RESET (位控复位)
    KEY_4 = auto()    # LOCO (行走模式)
    KEY_5 = auto()    # SKILL_1 (舞蹈)
    KEY_6 = auto()    # SKILL_2 (武术)
    KEY_7 = auto()    # SKILL_3 (武术2)
    KEY_8 = auto()    # SKILL_4 (踢腿)
    KEY_9 = auto()    # 退出程序
    KEY_T = auto()    # SKILL_5 (MotionTracking)

class KeyBoard:
    def __init__(self):
        self._keys_pressed = set()
        self._keys_just_released = set()
        self._current_keys = set()
        
        # 速度值
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_yaw = 0.0
        
        # 速度增量
        self._speed_scale = 1.0
        
        self._running = True
        self._lock = threading.Lock()
        
        if USE_PYNPUT:
            self._listener = pynput_keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self._listener.start()
        
        self._print_help()
    
    def _print_help(self):
        print("\n" + "="*50)
        print("键盘控制已启动 (不与MuJoCo viewer冲突):")
        print("="*50)
        print("  移动控制:")
        print("    ↑/↓:  前进/后退")
        print("    ←/→:  左转/右转")
        print("-"*50)
        print("  模式切换 (数字键 2-8):")
        print("    2:    阻尼保护模式 (PASSIVE)")
        print("    3:    位控复位 (POS_RESET)")
        print("    4:    行走模式 (LOCO)")
        print("    5:    舞蹈 (SKILL_1)")
        print("    6:    武术 (SKILL_2)")
        print("    7:    武术2 (SKILL_3)")
        print("    8:    踢腿 (SKILL_4)")
        print("    T:    MotionTracking (SKILL_5)")
        print("-"*50)
        print("    9:    退出程序")
        print("="*50)
        print("MuJoCo viewer 快捷键仍然有效:")
        print("  BACKSPACE: 重置机器人位置")
        print("  SPACE: 暂停/继续仿真")
        print("="*50 + "\n")
    
    def _pynput_to_key(self, key) -> KeyboardKey:
        """将pynput按键转换为KeyboardKey"""
        try:
            # 方向键
            if key == pynput_keyboard.Key.up:
                return KeyboardKey.UP
            elif key == pynput_keyboard.Key.down:
                return KeyboardKey.DOWN
            elif key == pynput_keyboard.Key.left:
                return KeyboardKey.LEFT
            elif key == pynput_keyboard.Key.right:
                return KeyboardKey.RIGHT
            
            # 数字键
            if hasattr(key, 'char') and key.char:
                char_map = {
                    '2': KeyboardKey.KEY_2,
                    '3': KeyboardKey.KEY_3,
                    '4': KeyboardKey.KEY_4,
                    '5': KeyboardKey.KEY_5,
                    '6': KeyboardKey.KEY_6,
                    '7': KeyboardKey.KEY_7,
                    '8': KeyboardKey.KEY_8,
                    '9': KeyboardKey.KEY_9,
                    't': KeyboardKey.KEY_T,
                }
                return char_map.get(key.char.lower())
        except AttributeError:
            pass
        return None
    
    def _on_press(self, key):
        """按键按下回调"""
        mapped_key = self._pynput_to_key(key)
        if mapped_key:
            with self._lock:
                self._current_keys.add(mapped_key)
    
    def _on_release(self, key):
        """按键释放回调"""
        mapped_key = self._pynput_to_key(key)
        if mapped_key:
            with self._lock:
                self._current_keys.discard(mapped_key)
                self._keys_just_released.add(mapped_key)
    
    def update(self):
        """更新键盘状态，应每帧调用"""
        with self._lock:
            # 更新按下的键
            self._keys_pressed = self._current_keys.copy()
            
            # 更新速度值
            self._vel_x = 0.0
            self._vel_y = 0.0
            self._vel_yaw = 0.0
            
            if KeyboardKey.UP in self._keys_pressed:
                self._vel_x = self._speed_scale
            if KeyboardKey.DOWN in self._keys_pressed:
                self._vel_x = -self._speed_scale
            if KeyboardKey.LEFT in self._keys_pressed:
                self._vel_yaw = self._speed_scale
            if KeyboardKey.RIGHT in self._keys_pressed:
                self._vel_yaw = -self._speed_scale
    
    def is_key_pressed(self, key: KeyboardKey) -> bool:
        """检查按键是否被按下"""
        return key in self._keys_pressed
    
    def is_key_released(self, key: KeyboardKey) -> bool:
        """检查按键是否刚被释放（只返回一次True）"""
        with self._lock:
            if key in self._keys_just_released:
                self._keys_just_released.discard(key)
                return True
        return False
    
    def get_velocity(self):
        """获取速度命令 (x, y, yaw)"""
        return self._vel_x, self._vel_y, self._vel_yaw
    
    def stop(self):
        """停止键盘监听"""
        self._running = False
        if USE_PYNPUT and hasattr(self, '_listener'):
            self._listener.stop()
    
    def __del__(self):
        self.stop()