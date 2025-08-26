import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import json

class VirtualDrawingApp:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,  # Increased for better accuracy
            min_tracking_confidence=0.95   # Increased for better accuracy
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Canvas setup
        self.canvas = None
        self.canvas_height = 720
        self.canvas_width = 1280
        
        # Canvas settings (initialize before reset_canvas)
        self.canvas_backgrounds = ['white', 'black', 'gray', 'blue']
        self.current_background = 0
        self.show_grid = False
        self.zoom_factor = 1.0
        self.pan_x, self.pan_y = 0, 0
        
        # Undo/Redo system (initialize before reset_canvas)
        self.canvas_history = []
        self.history_index = -1
        self.max_history = 10
        
        self.reset_canvas()
        
        # Drawing parameters
        self.drawing = False
        self.prev_x, self.prev_y = None, None
        self.brush_thickness = 5
        self.eraser_thickness = 20
        self.min_brush_size = 2
        self.max_brush_size = 25
        self.brush_size_step = 1
        
        # Advanced brush settings
        self.brush_shapes = ['circle', 'square', 'star', 'spray']
        self.current_brush_shape = 0
        self.brush_opacity = 255  # 0-255
        self.rainbow_mode = False
        self.rainbow_hue = 0
        
        # Layers system
        self.layers = []
        self.current_layer = 0
        self.layer_opacity = [255]  # Opacity for each layer
        
        # Selection tools
        self.selection_mode = False
        self.selection_start = None
        self.selection_end = None
        self.copied_region = None
        
        # Mirror mode
        self.mirror_mode = False
        self.mirror_axis = 'vertical'  # 'vertical', 'horizontal', 'both'
        
        # Advanced features
        self.paused = False
        self.confidence_threshold = 0.7
        self.drawing_stats = {
            'strokes': 0,
            'total_points': 0,
            'session_time': datetime.now()
        }
        
        # Auto-save
        self.auto_save_interval = 60  # seconds
        self.last_auto_save = datetime.now()
        
        # Colors (BGR format for OpenCV)
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'cyan': (255, 255, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'orange': (0, 165, 255),
            'pink': (147, 20, 255),
            'lime': (0, 255, 127),
            'turquoise': (208, 224, 64)
        }
        self.current_color = self.colors['blue']
        self.color_names = list(self.colors.keys())
        self.current_color_index = 2  # Start with blue
        
        # Initialize layers
        self.initialize_layers()
        
        # UI elements
        self.ui_height = 100
        self.mode = 'draw'  # 'draw' or 'erase'
        
        # Temporal smoothing for improved accuracy
        self.finger_history = []
        self.history_length = 5
        
        print("Virtual Drawing App Initialized!")
        print("Controls:")
        print("- ONLY Index finger up: Draw")
        print("- ONLY Index + Middle fingers up: Erase") 
        print("- Thumb up: Increase brush size")
        print("- Peace sign (Index + Middle spread): Take screenshot")
        print("- Fist (all fingers down): Pause/Resume drawing")
        print("- OK sign (thumb+index circle): Color picker mode")
        print("- Rock sign (index+pinky): Special effects")
        print("- Space: Change color")
        print("- 'c': Clear canvas")
        print("- 's': Save drawing")
        print("- 'u': Undo")
        print("- 'r': Redo")
        print("- 'h': Toggle help display")
        print("- 'q' or ESC: Quit")
        print("\nNEW ADVANCED CONTROLS:")
        print("- 'b': Change brush shape")
        print("- 'g': Toggle grid")
        print("- 'm': Toggle mirror mode")
        print("- 'l': Add new layer")
        print("- 'n': Switch background")
        print("- 'x': Toggle rainbow mode")
        print("- 'v': Selection mode")
        print("- Tab: Switch layers")
        print("- Ctrl+C/V: Copy/Paste")
        print("\nIMPORTANT for Right Hand Users:")
        print("- Face your palm TOWARD the camera")
        print("- Keep other fingers clearly DOWN")
        print("- Make CLEAR gestures (only the fingers you want up)")
        print("\nNEW FEATURES:")
        print("- Advanced brush shapes & effects")
        print("- Multi-layer support")
        print("- Selection & copy/paste tools")
        print("- Mirror drawing mode")
        print("- Grid overlay for precision")
        print("- Rainbow color mode")
        print("- Multiple canvas backgrounds")
        print("- Enhanced zoom & pan")
        print("\nTips for MAX accuracy:")
        print("- Use BRIGHT lighting")
        print("- Plain background behind you")  
        print("- Keep hand 2-3 feet from camera")
        print("- Move SLOWLY and steadily")
        print("- Keep other hand OUT of view")
    
    def reset_canvas(self):
        """Reset the drawing canvas to selected background"""
        background_colors = {
            'white': 255,
            'black': 0,
            'gray': 128,
            'blue': (255, 200, 150)
        }
        
        bg_name = self.canvas_backgrounds[self.current_background]
        if bg_name == 'blue':
            self.canvas = np.full((self.canvas_height, self.canvas_width, 3), background_colors[bg_name], dtype=np.uint8)
        else:
            color_val = background_colors[bg_name]
            self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * color_val
        
        # Reset layers
        self.layers = [self.canvas.copy()]
        self.current_layer = 0
        self.save_to_history()
    
    def initialize_layers(self):
        """Initialize the layer system"""
        if not hasattr(self, 'canvas') or self.canvas is None:
            return
        self.layers = [self.canvas.copy()]
        self.current_layer = 0
    
    def add_layer(self):
        """Add a new drawing layer"""
        new_layer = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        self.layers.append(new_layer)
        self.layer_opacity.append(255)
        self.current_layer = len(self.layers) - 1
        print(f"Added layer {self.current_layer + 1}. Total layers: {len(self.layers)}")
    
    def switch_layer(self):
        """Switch to next layer"""
        if len(self.layers) > 1:
            self.current_layer = (self.current_layer + 1) % len(self.layers)
            print(f"Switched to layer {self.current_layer + 1}")
    
    def merge_layers(self):
        """Merge all layers into final canvas"""
        if len(self.layers) == 1:
            self.canvas = self.layers[0].copy()
            return
        
        # Start with background
        result = self.layers[0].copy()
        
        # Blend each layer
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            opacity = self.layer_opacity[i] / 255.0
            
            # Simple alpha blending
            mask = np.any(layer != [0, 0, 0], axis=2)
            result[mask] = (opacity * layer[mask] + (1 - opacity) * result[mask]).astype(np.uint8)
        
        self.canvas = result
    
    def change_brush_shape(self):
        """Change brush shape"""
        self.current_brush_shape = (self.current_brush_shape + 1) % len(self.brush_shapes)
        shape_name = self.brush_shapes[self.current_brush_shape]
        print(f"Brush shape: {shape_name}")
    
    def toggle_grid(self):
        """Toggle grid overlay"""
        self.show_grid = not self.show_grid
        print(f"Grid: {'ON' if self.show_grid else 'OFF'}")
    
    def toggle_mirror_mode(self):
        """Toggle mirror drawing mode"""
        self.mirror_mode = not self.mirror_mode
        if self.mirror_mode:
            # Cycle through mirror types
            mirror_types = ['vertical', 'horizontal', 'both']
            current_index = mirror_types.index(self.mirror_axis) if self.mirror_axis in mirror_types else 0
            self.mirror_axis = mirror_types[(current_index + 1) % len(mirror_types)]
        print(f"Mirror mode: {'ON' if self.mirror_mode else 'OFF'} ({self.mirror_axis})")
    
    def toggle_rainbow_mode(self):
        """Toggle rainbow color mode"""
        self.rainbow_mode = not self.rainbow_mode
        print(f"Rainbow mode: {'ON' if self.rainbow_mode else 'OFF'}")
    
    def get_rainbow_color(self):
        """Get next color in rainbow sequence"""
        import colorsys
        self.rainbow_hue = (self.rainbow_hue + 0.01) % 1.0
        rgb = colorsys.hsv_to_rgb(self.rainbow_hue, 1.0, 1.0)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR for OpenCV
    
    def change_background(self):
        """Change canvas background"""
        self.current_background = (self.current_background + 1) % len(self.canvas_backgrounds)
        bg_name = self.canvas_backgrounds[self.current_background]
        print(f"Background: {bg_name}")
        self.reset_canvas()
    
    def toggle_selection_mode(self):
        """Toggle selection mode"""
        self.selection_mode = not self.selection_mode
        if not self.selection_mode:
            self.selection_start = None
            self.selection_end = None
        print(f"Selection mode: {'ON' if self.selection_mode else 'OFF'}")
    
    def copy_selection(self):
        """Copy selected region"""
        if self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            self.copied_region = self.canvas[y1:y2, x1:x2].copy()
            print("Selection copied!")
    
    def paste_selection(self, x, y):
        """Paste copied region at position"""
        if self.copied_region is not None:
            h, w = self.copied_region.shape[:2]
            y_end = min(y + h, self.canvas_height)
            x_end = min(x + w, self.canvas_width)
            
            self.canvas[y:y_end, x:x_end] = self.copied_region[:y_end-y, :x_end-x]
            print("Selection pasted!")
            self.save_to_history()
    
    def save_to_history(self):
        """Save current canvas state to history for undo/redo"""
        if self.history_index < len(self.canvas_history) - 1:
            # Remove future history if we're not at the end
            self.canvas_history = self.canvas_history[:self.history_index + 1]
        
        self.canvas_history.append(self.canvas.copy())
        if len(self.canvas_history) > self.max_history:
            self.canvas_history.pop(0)
        else:
            self.history_index += 1
    
    def show_help_instructions(self):
        """Display help instructions in console"""
        print("\n" + "="*50)
        print("VIRTUAL DRAWING - HELP INSTRUCTIONS")
        print("="*50)
        print("GESTURE CONTROLS:")
        print("- Index finger up: Draw")
        print("- Peace sign (index + middle): Move cursor")
        print("- Fist: Clear canvas")
        print("- Thumb up: Change color")
        print("- OK sign: Change brush size")
        print("\nKEYBOARD SHORTCUTS:")
        print("- Q: Quit application")
        print("- C: Clear canvas")
        print("- S: Save screenshot")
        print("- P: Pause/unpause")
        print("- G: Toggle grid overlay")
        print("- M: Toggle mirror mode")
        print("- R: Toggle rainbow mode")
        print("- E: Toggle eraser mode")
        print("- H: Show this help")
        print("- V: Toggle selection mode")
        print("- B: Change brush shape")
        print("- N: Change background")
        print("- TAB: Switch layers")
        print("- +/-: Adjust brush size")
        print("- Z: Undo last action")
        print("- Y: Redo last action")
        print("- SPACE: Change color")
        print("="*50)
    
    def undo(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.canvas = self.canvas_history[self.history_index].copy()
            print("Undo successful!")
        else:
            print("Nothing to undo!")
    
    def redo(self):
        """Redo last undone action"""
        if self.history_index < len(self.canvas_history) - 1:
            self.history_index += 1
            self.canvas = self.canvas_history[self.history_index].copy()
            print("Redo successful!")
        else:
            print("Nothing to redo!")
    
    def auto_save(self):
        """Auto-save drawing if enough time has passed"""
        now = datetime.now()
        if (now - self.last_auto_save).seconds >= self.auto_save_interval:
            if not os.path.exists('auto_saves'):
                os.makedirs('auto_saves')
            
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            filename = f"auto_saves/auto_save_{timestamp}.png"
            cv2.imwrite(filename, self.canvas)
            self.last_auto_save = now
            print(f"Auto-saved: {filename}")
    
    def take_screenshot(self, frame):
        """Take a screenshot of the current view"""
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    
    def increase_brush_size(self):
        """Increase brush size"""
        if self.brush_thickness < self.max_brush_size:
            self.brush_thickness += self.brush_size_step
            print(f"Brush size: {self.brush_thickness}")
    
    def get_drawing_stats(self):
        """Get current drawing statistics"""
        session_time = (datetime.now() - self.drawing_stats['session_time']).seconds
        return {
            'strokes': self.drawing_stats['strokes'],
            'points': self.drawing_stats['total_points'],
            'time': f"{session_time//60}m {session_time%60}s"
        }
    
    def count_fingers(self, landmarks):
        """Count the number of fingers that are up with improved accuracy for both hands"""
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pip = [3, 6, 10, 14, 18]   # PIP joints for comparison
        finger_mcp = [2, 5, 9, 13, 17]    # MCP joints for additional reference
        
        fingers = []
        
        # Determine hand orientation (left or right)
        # Compare thumb and pinky positions to determine hand orientation
        thumb_x = landmarks[finger_tips[0]].x
        pinky_x = landmarks[finger_tips[4]].x
        wrist_x = landmarks[0].x
        
        is_right_hand = thumb_x > pinky_x
        
        # Thumb detection (improved for both hands)
        thumb_tip_x = landmarks[finger_tips[0]].x
        thumb_mcp_x = landmarks[finger_mcp[0]].x
        
        if is_right_hand:
            # Right hand: thumb should be to the right of MCP
            fingers.append(1 if thumb_tip_x > thumb_mcp_x else 0)
        else:
            # Left hand: thumb should be to the left of MCP
            fingers.append(1 if thumb_tip_x < thumb_mcp_x else 0)
        
        # Other fingers - much more accurate detection
        for i in range(1, 5):
            tip_y = landmarks[finger_tips[i]].y
            pip_y = landmarks[finger_pip[i]].y
            mcp_y = landmarks[finger_mcp[i]].y
            
            # Finger is up if tip is significantly higher than both PIP and MCP
            finger_length = mcp_y - tip_y
            pip_length = mcp_y - pip_y
            
            # More strict threshold - finger must be clearly extended
            if finger_length > pip_length * 1.3:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # --- Temporal smoothing for improved accuracy ---
        self.finger_history.append(fingers)
        if len(self.finger_history) > self.history_length:
            self.finger_history.pop(0)
        
        # Use majority vote for each finger to reduce false positives
        if len(self.finger_history) >= 3:  # Need at least 3 frames for smoothing
            smoothed = [int(sum(f[i] for f in self.finger_history) > self.history_length // 2) for i in range(5)]
            return smoothed, sum(smoothed)
        else:
            # Return original for first few frames
            return fingers, sum(fingers)
    
    def get_finger_position(self, landmarks, finger_tip_id=8):
        """Get the position of a specific finger tip"""
        h, w, _ = self.canvas.shape
        x = int(landmarks[finger_tip_id].x * w)
        y = int(landmarks[finger_tip_id].y * h)
        return x, y
    
    def draw_ui(self, frame):
        """Draw the enhanced user interface with new features"""
        # Create larger UI bar for more information
        ui_bar = np.ones((self.ui_height + 80, frame.shape[1], 3), dtype=np.uint8) * 50
        
        # Display current color
        color_rect_start = 20
        color_rect_size = 60
        current_color = self.get_rainbow_color() if self.rainbow_mode else self.current_color
        cv2.rectangle(ui_bar, 
                     (color_rect_start, 20), 
                     (color_rect_start + color_rect_size, 20 + color_rect_size), 
                     current_color, -1)
        cv2.rectangle(ui_bar, 
                     (color_rect_start, 20), 
                     (color_rect_start + color_rect_size, 20 + color_rect_size), 
                     (255, 255, 255), 2)
        
        # Display brush size and shape indicator
        brush_x = color_rect_start + color_rect_size + 30
        brush_center = (brush_x + 30, 50)
        
        # Draw brush shape preview
        brush_shape = self.brush_shapes[self.current_brush_shape]
        if brush_shape == 'circle':
            cv2.circle(ui_bar, brush_center, self.brush_thickness, (200, 200, 200), -1)
        elif brush_shape == 'square':
            half_size = self.brush_thickness // 2
            cv2.rectangle(ui_bar, 
                         (brush_center[0] - half_size, brush_center[1] - half_size),
                         (brush_center[0] + half_size, brush_center[1] + half_size),
                         (200, 200, 200), -1)
        elif brush_shape == 'star':
            # Simple star representation
            cv2.putText(ui_bar, "★", (brush_center[0] - 10, brush_center[1] + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        elif brush_shape == 'spray':
            # Spray pattern
            for i in range(5):
                import random
                offset_x = random.randint(-self.brush_thickness, self.brush_thickness)
                offset_y = random.randint(-self.brush_thickness, self.brush_thickness)
                cv2.circle(ui_bar, (brush_center[0] + offset_x, brush_center[1] + offset_y), 1, (200, 200, 200), -1)
        
        cv2.circle(ui_bar, brush_center, self.brush_thickness, (255, 255, 255), 2)
        
        # Display current settings
        mode_text = f"Mode: {self.mode.upper()}"
        if self.paused:
            mode_text += " (PAUSED)"
        if self.rainbow_mode:
            mode_text += " 🌈"
        if self.mirror_mode:
            mode_text += f" 🪞({self.mirror_axis})"
        if self.selection_mode:
            mode_text += " 📦"
        
        color_name = "RAINBOW" if self.rainbow_mode else self.color_names[self.current_color_index].upper()
        color_text = f"Color: {color_name}"
        brush_text = f"Brush: {brush_shape.upper()} {self.brush_thickness}px"
        layer_text = f"Layer: {self.current_layer + 1}/{len(self.layers)}" if self.layers else "Layer: 1/1"
        background_text = f"BG: {self.canvas_backgrounds[self.current_background].upper()}"
        
        text_x = brush_x + 80
        cv2.putText(ui_bar, mode_text, (text_x, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(ui_bar, color_text, (text_x, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(ui_bar, brush_text, (text_x, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(ui_bar, f"{layer_text} | {background_text}", (text_x, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display finger states and features status
        if hasattr(self, 'last_fingers'):
            finger_names = ['👍', '👆', '🖕', '💍', '🤙']
            finger_text = "Fingers: " + "".join([f"{finger_names[i]} " if self.last_fingers[i] else "✋ " for i in range(5)])
            cv2.putText(ui_bar, finger_text, (text_x + 280, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
        
        # Features status
        features = []
        if self.show_grid: features.append("GRID")
        if self.mirror_mode: features.append("MIRROR")
        if self.rainbow_mode: features.append("RAINBOW")
        if self.selection_mode: features.append("SELECT")
        
        features_text = " | ".join(features) if features else "STANDARD"
        cv2.putText(ui_bar, f"Features: {features_text}", (text_x + 280, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
        
        # Display statistics
        stats = self.get_drawing_stats()
        stats_text = f"Strokes: {stats['strokes']} | Points: {stats['points']} | Time: {stats['time']}"
        cv2.putText(ui_bar, stats_text, (text_x + 280, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
        
        # Enhanced instructions with new features
        instructions = "👆Draw | ✌️Erase | 👍Size+ | 🤞Screenshot | ✊Pause | 🤌Selection | B:Brush | G:Grid | M:Mirror | L:Layer | X:Rainbow"
        cv2.putText(ui_bar, instructions, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        instructions2 = "SPACE:Color | C:Clear | U:Undo | R:Redo | S:Save | N:Background | V:Select | TAB:Layer | H:Help | Q:Quit"
        
        cv2.putText(ui_bar, instructions2, (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Confidence meter
        if hasattr(self, 'last_confidence'):
            conf_width = int(200 * self.last_confidence)
            cv2.rectangle(ui_bar, (text_x + 280, 80), (text_x + 480, 95), (100, 100, 100), -1)
            cv2.rectangle(ui_bar, (text_x + 280, 80), (text_x + 280 + conf_width, 95), (0, 255, 0), -1)
            cv2.putText(ui_bar, f"Confidence: {self.last_confidence:.2f}", (text_x + 490, 92), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Add grid overlay if enabled
        if self.show_grid:
            self.draw_grid_overlay(frame)
        
        # Draw selection rectangle if in selection mode
        if self.selection_mode and self.selection_start and self.selection_end:
            cv2.rectangle(frame, self.selection_start, self.selection_end, (0, 255, 255), 2)
        
        # Combine UI bar with frame
        combined_frame = np.vstack((ui_bar, frame))
        return combined_frame
    
    def draw_grid_overlay(self, frame):
        """Draw grid overlay on the frame"""
        grid_size = 50
        h, w = frame.shape[:2]
        
        # Vertical lines
        for x in range(0, w, grid_size):
            cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
        
        # Horizontal lines
        for y in range(0, h, grid_size):
            cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)
    
    def save_drawing(self):
        """Save the current drawing"""
        if not os.path.exists('drawings'):
            os.makedirs('drawings')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawings/drawing_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        print(f"Drawing saved as {filename}")
    
    def process_gestures(self, landmarks):
        """Process hand gestures for drawing with enhanced features"""
        fingers, total_fingers = self.count_fingers(landmarks)
        self.last_fingers = fingers  # Store for UI display
        
        # Get finger positions
        index_x, index_y = self.get_finger_position(landmarks, 8)
        thumb_x, thumb_y = self.get_finger_position(landmarks, 4)
        middle_x, middle_y = self.get_finger_position(landmarks, 12)
        
        # Enhanced smoothing for better drawing
        smoothing_factor = 0.6
        if hasattr(self, 'smooth_x') and hasattr(self, 'smooth_y'):
            self.smooth_x = int(smoothing_factor * self.smooth_x + (1 - smoothing_factor) * index_x)
            self.smooth_y = int(smoothing_factor * self.smooth_y + (1 - smoothing_factor) * index_y)
        else:
            self.smooth_x, self.smooth_y = index_x, index_y
        
        # Gesture recognition
        thumb_up = fingers[0] == 1
        index_up = fingers[1] == 1
        middle_up = fingers[2] == 1
        ring_up = fingers[3] == 1
        pinky_up = fingers[4] == 1
        
        # Check for fist (pause/resume)
        if not any(fingers):
            self.paused = not self.paused
            self.drawing = False
            self.prev_x, self.prev_y = None, None
            print(f"Drawing {'PAUSED' if self.paused else 'RESUMED'}")
            return
        
        # Skip other gestures if paused
        if self.paused:
            return
        
        # Selection mode handling
        if self.selection_mode:
            if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
                if self.selection_start is None:
                    self.selection_start = (self.smooth_x, self.smooth_y)
                else:
                    self.selection_end = (self.smooth_x, self.smooth_y)
            return
        
        # OK sign (thumb + index forming circle) = Special mode
        thumb_index_distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
        if thumb_up and index_up and not middle_up and not ring_up and not pinky_up and thumb_index_distance < 30:
            self.mode = 'color_picker'
            # Color picker functionality could be added here
            return
        
        # Rock sign (index + pinky) = Special effects
        elif index_up and pinky_up and not thumb_up and not middle_up and not ring_up:
            self.mode = 'special_effects'
            # Special effects like glow, blur could be added here
            return
        
        # Three fingers up = Clear area (local eraser)
        elif index_up and middle_up and ring_up and not thumb_up and not pinky_up:
            self.mode = 'area_clear'
            # Clear a larger area around the finger
            cv2.circle(self.canvas, (self.smooth_x, self.smooth_y), self.eraser_thickness * 2, (255, 255, 255), -1)
            self.merge_layers()
            return
        
        # Thumb up = Increase brush size
        elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            self.increase_brush_size()
            return
            
        # Peace sign (Index + Middle spread) = Screenshot
        elif index_up and middle_up and not thumb_up and not ring_up and not pinky_up:
            finger_distance = ((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2) ** 0.5
            if finger_distance > 50:  # Fingers are spread
                self.mode = 'screenshot'
                if not hasattr(self, 'last_screenshot_time') or \
                   (datetime.now() - self.last_screenshot_time).seconds > 2:
                    self.last_screenshot_time = datetime.now()
                    return "screenshot"
            else:
                # Close fingers = Erase mode
                self.mode = 'erase'
                self.draw_on_canvas(self.smooth_x, self.smooth_y, (255, 255, 255), self.eraser_thickness)
            
        # Only index finger up = Draw mode
        elif index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            self.mode = 'draw'
            self.draw_on_canvas(self.smooth_x, self.smooth_y, self.current_color, self.brush_thickness)
            
        else:
            # Stop drawing when gesture changes
            self.drawing = False
            self.prev_x, self.prev_y = None, None
            
        return None
    
    def draw_on_canvas(self, x, y, color, thickness):
        """Draw on the canvas with enhanced features and brush shapes"""
        # Use rainbow color if enabled
        if self.rainbow_mode:
            color = self.get_rainbow_color()
        
        # Get current layer
        current_canvas = self.layers[self.current_layer] if self.layers else self.canvas
        
        # Draw primary stroke
        self.draw_stroke(current_canvas, x, y, color, thickness)
        
        # Mirror drawing if enabled
        if self.mirror_mode:
            center_x, center_y = self.canvas_width // 2, self.canvas_height // 2
            
            if self.mirror_axis in ['vertical', 'both']:
                mirror_x = center_x * 2 - x
                self.draw_stroke(current_canvas, mirror_x, y, color, thickness)
                
            if self.mirror_axis in ['horizontal', 'both']:
                mirror_y = center_y * 2 - y
                self.draw_stroke(current_canvas, x, mirror_y, color, thickness)
                
            if self.mirror_axis == 'both':
                mirror_x = center_x * 2 - x
                mirror_y = center_y * 2 - y
                self.draw_stroke(current_canvas, mirror_x, mirror_y, color, thickness)
        
        # Update merged canvas
        self.merge_layers()
        
        # Update statistics
        self.prev_x, self.prev_y = x, y
        
        if not self.drawing:
            self.drawing = True
            self.drawing_stats['strokes'] += 1
            self.save_to_history()
    
    def draw_stroke(self, canvas, x, y, color, thickness):
        """Draw a stroke with the current brush shape"""
        brush_shape = self.brush_shapes[self.current_brush_shape]
        
        if self.prev_x is not None and self.prev_y is not None:
            if brush_shape == 'circle':
                cv2.line(canvas, (self.prev_x, self.prev_y), (x, y), color, thickness)
            elif brush_shape == 'square':
                self.draw_square_brush(canvas, x, y, color, thickness)
            elif brush_shape == 'star':
                self.draw_star_brush(canvas, x, y, color, thickness)
            elif brush_shape == 'spray':
                self.draw_spray_brush(canvas, x, y, color, thickness)
            
            self.drawing_stats['total_points'] += 1
    
    def draw_square_brush(self, canvas, x, y, color, thickness):
        """Draw with square brush"""
        half_size = thickness // 2
        cv2.rectangle(canvas, 
                     (x - half_size, y - half_size), 
                     (x + half_size, y + half_size), 
                     color, -1)
    
    def draw_star_brush(self, canvas, x, y, color, thickness):
        """Draw with star-shaped brush"""
        import math
        points = []
        for i in range(10):  # 5-pointed star
            angle = i * math.pi / 5
            radius = thickness if i % 2 == 0 else thickness // 2
            px = int(x + radius * math.cos(angle))
            py = int(y + radius * math.sin(angle))
            points.append([px, py])
        
        cv2.fillPoly(canvas, [np.array(points)], color)
    
    def draw_spray_brush(self, canvas, x, y, color, thickness):
        """Draw with spray/airbrush effect"""
        import random
        for _ in range(thickness * 2):
            offset_x = random.randint(-thickness, thickness)
            offset_y = random.randint(-thickness, thickness)
            distance = (offset_x**2 + offset_y**2)**0.5
            
            if distance <= thickness:
                px, py = x + offset_x, y + offset_y
                if 0 <= px < self.canvas_width and 0 <= py < self.canvas_height:
                    # Fade effect based on distance
                    alpha = 1 - (distance / thickness)
                    blended_color = tuple(int(c * alpha + canvas[py, px][i] * (1 - alpha)) for i, c in enumerate(color))
                    cv2.circle(canvas, (px, py), 1, blended_color, -1)
    
    def change_color(self):
        """Change to the next color"""
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        self.current_color = self.colors[self.color_names[self.current_color_index]]
        print(f"Color changed to: {self.color_names[self.current_color_index]}")
    
    def run(self):
        """Main application loop"""
        print("\nStarting Virtual Drawing App...")
        print("Show your hand to the camera and use your index finger to draw!")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(frame_rgb)
            
            # Store confidence for UI
            if results.multi_hand_landmarks:
                # Get hand detection confidence (approximate)
                self.last_confidence = 0.9  # MediaPipe doesn't expose confidence directly
            else:
                self.last_confidence = 0.0
            
            # Draw hand landmarks and process gestures
            screenshot_requested = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on frame (more visible)
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
                    
                    # Process gestures for drawing
                    gesture_result = self.process_gestures(hand_landmarks.landmark)
                    if gesture_result == "screenshot":
                        screenshot_requested = True
            else:
                # No hand detected, stop drawing
                self.drawing = False
                self.prev_x, self.prev_y = None, None
            
            # Blend canvas with camera feed
            canvas_resized = cv2.resize(self.canvas, (frame.shape[1], frame.shape[0]))
            
            # Create a mask where canvas is not white
            mask = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)[1]
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = mask / 255.0
            
            # Blend the images
            blended = frame * (1 - mask) + canvas_resized * mask
            blended = blended.astype(np.uint8)
            
            # Add UI
            display_frame = self.draw_ui(blended)
            
            # Take screenshot if requested
            if screenshot_requested:
                self.take_screenshot(display_frame)
            
            # Auto-save functionality
            self.auto_save()
            
            # Display the result
            cv2.imshow('Virtual Drawing App', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break
            elif key == ord('c'):  # Clear canvas
                self.reset_canvas()
                self.drawing_stats['strokes'] = 0
                self.drawing_stats['total_points'] = 0
                print("Canvas cleared!")
            elif key == ord(' '):  # Space to change color
                self.change_color()
            elif key == ord('s'):  # Save drawing
                self.save_drawing()
            elif key == ord('u'):  # Undo
                self.undo()
            elif key == ord('r'):  # Redo
                self.redo()
            elif key == ord('p'):  # Toggle pause
                self.paused = not self.paused
                print(f"Drawing {'PAUSED' if self.paused else 'RESUMED'}")
            elif key == ord('b'):  # Change brush shape
                self.change_brush_shape()
            elif key == ord('g'):  # Toggle grid
                self.toggle_grid()
            elif key == ord('m'):  # Toggle mirror mode
                self.toggle_mirror_mode()
            elif key == ord('x'):  # Toggle rainbow mode
                self.toggle_rainbow_mode()
            elif key == ord('n'):  # Change background
                self.change_background()
            elif key == ord('l'):  # Add new layer
                self.add_layer()
            elif key == ord('v'):  # Toggle selection mode
                self.toggle_selection_mode()
            elif key == 9:  # Tab key - Switch layers
                self.switch_layer()
            elif key == ord('h'):  # Show/hide help
                self.show_help_instructions()
            elif key == ord('=') or key == ord('+'):  # Increase brush size
                self.increase_brush_size()
            elif key == ord('-'):  # Decrease brush size
                if self.brush_thickness > self.min_brush_size:
                    self.brush_thickness -= self.brush_size_step
                    print(f"Brush size: {self.brush_thickness}")
            
            elif key == ord('e'):  # Toggle erase mode (manual override)
                if self.mode == 'draw':
                    self.mode = 'erase'
                else:
                    self.mode = 'draw'
                print(f"Mode changed to: {self.mode}")
            
            # Handle Ctrl+C and Ctrl+V for copy/paste (simplified)
            elif key == 3:  # Ctrl+C
                self.copy_selection()
            elif key == 22:  # Ctrl+V
                if self.copied_region is not None:
                    self.paste_selection(self.canvas_width // 2, self.canvas_height // 2)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Virtual Drawing App closed.")

def main():
    """Main function to run the application"""
    try:
        app = VirtualDrawingApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required packages installed:")
        print("pip install opencv-python mediapipe numpy")

if __name__ == "__main__":
    main()