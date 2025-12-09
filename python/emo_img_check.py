# emo_img_check.py
"""
얼굴 감정 분석 결과 검증 도구
판별하려는 감정과 판별된 감정을 비교하여 일치 여부를 보여주고,
불일치하는 경우 사용자가 직접 감정을 판별하여 기록할 수 있음.
[require install] pip install dearpygui pandas numpy pillow
"""

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
import os
import json
import traceback
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image


class Config:
    """설정 상수 관리"""
    TEXTURE_WIDTH = 500
    TEXTURE_HEIGHT = 500
    STATE_FILE = "validator_state.json"
    MISMATCH_FOLDER = "mismatch"
    IMAGE_PREFIX = "cropped_"
    IMAGE_EXTENSIONS = ".jpg", ".jpeg", ".png"
    
    EMOTIONS = [
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "neutral",
    ]


class EmotionValidator:
    """감정 분류 결과 검증 도구"""

    def __init__(self):
        # 데이터 관련
        self.csv_file: Optional[str] = None
        self.image_folder: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.image_files: List[str] = []
        self.current_idx: int = 0

        # 감정 관련
        self.selected_emotion: str = "neutral"
        self.user_judgment: Optional[str] = None

        # 텍스처 관리
        self.texture_tag = "image_texture"
        self.image_widget_tag = "image_display"
        self.texture_created = False

        # UI 초기화
        self.setup_ui()
        self.load_state()

    # ==================== UI SETUP ====================
    def setup_ui(self):
        """메인 UI 구성"""
        with dpg.window(
            label="Emotion Validation Tool",
            width=1200,
            height=800,
            tag="main_window",
        ):
            self._setup_top_controls()
            self._setup_main_area()

    def _setup_top_controls(self):
        """상단 컨트롤 영역 - 이제 비어있음"""
        # ✅ 상단 버튼들을 제거하고 아래로 이동했으므로 비워둠
        pass

    def _setup_main_area(self):
        """메인 영역: 이미지 + 결과"""
        with dpg.group(horizontal=True):
            self._setup_image_area()
            self._setup_result_area()

    def _setup_image_area(self):
        """이미지 표시 영역"""
        with dpg.child_window(width=550, height=650, tag="image_area"):
            dpg.add_text(
                "Select image folder and CSV file, then click Start.",
                tag="image_text",
            )

    def _setup_result_area(self):
        """결과 및 컨트롤 영역"""
        with dpg.child_window(width=600, height=650, tag="result_area"):
            # ✅ Emotion Scores 타이틀 추가 (노란색)
            dpg.add_text("Emotion Scores:", tag="score_title", color=[255, 255, 0])
            dpg.add_text("waiting...", tag="score_text", wrap=580)
            
            dpg.add_text("Result: -", tag="match_text", color=[255, 255, 255])
            dpg.add_text("Progress: 0/0 (0%)", tag="progress_text")

            dpg.add_separator()
            
            # Target emotion 선택
            self._setup_target_emotion_selector()
            
            dpg.add_separator()
            
            # User Judgment + Save State (4x2 행렬)
            self._setup_judgment_area()

            dpg.add_separator()
            
            # Prev/Next 버튼
            self._setup_navigation_buttons()
            
            dpg.add_separator()
            
            # ✅ Image Folder, CSV File, Start 버튼들을 여기로 이동
            self._setup_file_controls()

    def _setup_target_emotion_selector(self):
        """Target emotion 선택 영역 (크게)"""
        dpg.add_text("Target emotion:", color=[255, 200, 0])
        dpg.add_combo(
            items=Config.EMOTIONS,
            default_value=self.selected_emotion,
            callback=self.on_emotion_change,
            tag="target_emotion_combo_large",
            width=250,
            height_mode=dpg.mvComboHeight_Large,
        )

    def _setup_judgment_area(self):
        """사용자 판별 영역 (4x2 행렬 + Save State)"""
        # ✅ User Judgment 타이틀과 상태를 같은 줄에 표시
        with dpg.group(horizontal=True):
            dpg.add_text("User Judgment (for mismatches):", color=[255, 255, 0])
            dpg.add_text("", tag="judgment_status", color=[0, 255, 0])
        
        # ✅ 4x2 행렬로 버튼 배치 (마지막에 Save State)
        # 첫 번째 행: anger, disgust, fear, happiness
        with dpg.group(horizontal=True):
            for emotion in Config.EMOTIONS[:4]:
                dpg.add_button(
                    label=emotion.title(),
                    callback=self.on_user_judgment,
                    user_data=emotion,
                    width=130,
                    height=40,
                    tag=f"judge_{emotion}",
                )
        
        # 두 번째 행: sadness, surprise, neutral, Save State
        with dpg.group(horizontal=True):
            for emotion in Config.EMOTIONS[4:]:
                dpg.add_button(
                    label=emotion.title(),
                    callback=self.on_user_judgment,
                    user_data=emotion,
                    width=130,
                    height=40,
                    tag=f"judge_{emotion}",
                )
            
            # ✅ Save State 버튼 (다른 색상)
            with dpg.theme() as save_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(
                        dpg.mvThemeCol_Button,
                        (50, 100, 150, 255),  # 파란색 계열
                        category=dpg.mvThemeCat_Core
                    )
                    dpg.add_theme_color(
                        dpg.mvThemeCol_ButtonHovered,
                        (70, 120, 170, 255),
                        category=dpg.mvThemeCat_Core
                    )
                    dpg.add_theme_color(
                        dpg.mvThemeCol_ButtonActive,
                        (40, 90, 140, 255),
                        category=dpg.mvThemeCat_Core
                    )
            
            save_btn = dpg.add_button(
                label="Save State",
                width=130,
                height=40,
                callback=self.save_current_and_state
            )
            dpg.bind_item_theme(save_btn, save_theme)

    def _setup_navigation_buttons(self):
        """네비게이션 버튼 (Prev/Next만)"""
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Prev", 
                width=280, 
                height=50, 
                callback=self.prev_image
            )
            dpg.add_button(
                label="Next", 
                width=280, 
                height=50, 
                callback=self.next_image
            )

    def _setup_file_controls(self):
        """파일 선택 컨트롤 (Prev/Next 아래로 이동)"""
        dpg.add_text("File Controls:", color=[255, 255, 0])
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Image Folder",
                width=180,
                height=40,
                callback=self.select_image_folder
            )
            dpg.add_button(
                label="CSV File",
                width=180,
                height=40,
                callback=self.select_csv
            )
        dpg.add_button(
            label="Start Analysis",
            width=370,
            height=40,
            callback=self.start_analysis
        )

    # ==================== FILE DIALOGS ====================
    def select_image_folder(self, sender, app_data, user_data=None):
        """이미지 폴더 선택 다이얼로그"""
        self._show_folder_dialog()

    def _show_folder_dialog(self):
        """폴더 선택 다이얼로그 표시"""
        if dpg.does_item_exist("folder_dialog"):
            dpg.delete_item("folder_dialog")

        with dpg.file_dialog(
            modal=True,
            directory_selector=True,
            show=False,
            callback=self.on_folder_select,
            cancel_callback=lambda s, a: None,
            tag="folder_dialog",
            width=700,
            height=400,
            default_path=os.path.expanduser("~"),
        ):
            dpg.add_file_extension(".*", color=(150, 150, 150, 255))

        dpg.show_item("folder_dialog")

    def select_csv(self, sender, app_data, user_data=None):
        """CSV 파일 선택 다이얼로그"""
        self._show_csv_dialog()

    def _show_csv_dialog(self):
        """CSV 선택 다이얼로그 표시"""
        if dpg.does_item_exist("csv_dialog"):
            dpg.delete_item("csv_dialog")

        with dpg.file_dialog(
            modal=True,
            directory_selector=False,
            show=False,
            callback=self.on_csv_select,
            cancel_callback=lambda s, a: None,
            tag="csv_dialog",
            width=700,
            height=400,
            default_path=self.image_folder or os.path.expanduser("~"),
        ):
            # ✅ *.csv가 아니라 .csv 형식으로 수정
            dpg.add_file_extension(".csv", color=(0, 255, 0, 255), custom_text="[CSV]")
            dpg.add_file_extension("", color=(150, 150, 150, 255), custom_text="[All Files]")

        dpg.show_item("csv_dialog")

    def on_folder_select(self, sender, app_data):
        """폴더 선택 후 처리"""
        folder = app_data.get("file_path_name")
        if not folder:
            return

        self.image_folder = folder
        self._load_image_files()
        self._ensure_mismatch_folder()
        
        self._update_ui_message(
            "image_text",
            f"Folder loaded: {len(self.image_files)} images\n{folder}"
        )

    def _load_image_files(self):
        """이미지 파일 목록 로드"""
        if not self.image_folder:
            return
            
        self.image_files = [
            f
            for f in os.listdir(self.image_folder)
            if f.startswith(Config.IMAGE_PREFIX)
            and f.lower().endswith(Config.IMAGE_EXTENSIONS)
        ]
        self.image_files.sort()

    def _ensure_mismatch_folder(self):
        """mismatch 폴더 생성"""
        if not self.image_folder:
            return
            
        mismatch_path = Path(self.image_folder) / Config.MISMATCH_FOLDER
        mismatch_path.mkdir(exist_ok=True)
        print(f"Mismatch folder ready: {mismatch_path}")

    def on_csv_select(self, sender, app_data):
        """CSV 파일 선택 후 처리"""
        path = app_data.get("file_path_name")
        if not path or not self._validate_csv_path(path):
            return

        try:
            self._load_csv_file(path)
        except Exception as e:
            self._show_error("score_text", f"Error loading CSV: {str(e)}")
            self.csv_file = None
            self.df = None

    def _validate_csv_path(self, path: str) -> bool:
        """CSV 경로 유효성 검사"""
        if not os.path.exists(path):
            self._show_error("score_text", f"Error: File does not exist\n{path}")
            return False
        if not path.lower().endswith(".csv"):
            self._show_error("score_text", "Error: Please select a CSV file")
            return False
        return True

    def _load_csv_file(self, path: str):
        """CSV 파일 로드 및 컬럼 초기화"""
        self.csv_file = path
        self.df = pd.read_csv(self.csv_file)

        # 필요한 컬럼 추가 및 정렬
        self._prepare_dataframe_columns()
        
        # ✅ 타이틀은 그대로 두고 본문만 업데이트
        self._update_ui_message(
            "score_text",
            f"CSV loaded: {len(self.df)} rows\n{os.path.basename(path)}"
        )

    def _prepare_dataframe_columns(self):
        """DataFrame 컬럼 준비 및 정렬"""
        # 필요한 컬럼 추가
        for col in ["emotion", "cmp_result"]:
            if col not in self.df.columns:
                self.df[col] = ""
        
        result_cols = ["cmp_result", "emotion"]
        other_cols = [c for c in self.df.columns if c not in result_cols]
        self.df = self.df[other_cols + result_cols]

        # 빈 문자열로 초기화 (NaN 방지)
        self.df["cmp_result"] = self.df["cmp_result"].fillna("").astype(str)
        self.df["emotion"] = self.df["emotion"].fillna("").astype(str)

    # ==================== MAIN FLOW ====================
    def start_analysis(self, sender, app_data, user_data=None):
        """분석 시작"""
        if not self._validate_start_conditions():
            self._show_error(
                "image_text",
                "Error: select image folder and CSV file first."
            )
            return
        self.load_image()

    def _validate_start_conditions(self) -> bool:
        """시작 조건 검증"""
        return all([
            self.image_folder,
            self.image_files,
            self.csv_file,
            self.df is not None
        ])

    def load_image(self):
        """현재 인덱스의 이미지 로드 및 표시"""
        if not self._check_image_availability():
            return

        filename = self.image_files[self.current_idx]
        img_path = os.path.join(self.image_folder, filename)

        print(f"Loading image: {filename} (index: {self.current_idx})")

        # 사용자 판별 초기화
        self._reset_user_judgment()

        # 이미지 표시 및 점수 표시
        try:
            if self._display_image(img_path, filename):
                self._display_scores(filename)
            else:
                print(f"Failed to display image: {filename}")
        except Exception as e:
            self._log_error("load_image", e)

        self.update_progress()

    def _check_image_availability(self) -> bool:
        """이미지 가용성 확인"""
        if not self.image_files:
            self._update_ui_message("image_text", "No images in selected folder.")
            return False

        if self.current_idx >= len(self.image_files):
            self._update_ui_message("image_text", "All images processed.")
            self.update_progress()
            return False

        return True

    def _reset_user_judgment(self):
        """사용자 판별 초기화"""
        self.user_judgment = None
        dpg.set_value("judgment_status", "")

    def _display_image(self, img_path: str, filename: str) -> bool:
        """고정 크기 텍스처를 사용한 이미지 표시"""
        try:
            # 이미지 로드 및 캔버스 생성
            canvas = self._create_image_canvas(img_path)
            
            # 텍스처 데이터 준비
            img_array = np.array(canvas, dtype=np.float32) / 255.0
            data = img_array.flatten().tolist()

            # 텍스처 생성 또는 업데이트
            self._update_texture(data)
            
            self._update_ui_message("image_text", filename)
            return True

        except Exception as e:
            self._log_error("display_image", e)
            self._show_error("image_text", f"Failed to load: {filename}")
            return False

    def _create_image_canvas(self, img_path: str) -> Image.Image:
        """이미지를 고정 크기 캔버스에 배치"""
        img = Image.open(img_path).convert("RGBA")
        img.thumbnail(
            (Config.TEXTURE_WIDTH, Config.TEXTURE_HEIGHT),
            Image.Resampling.LANCZOS
        )

        # 검은 배경 캔버스 생성
        canvas = Image.new(
            "RGBA",
            (Config.TEXTURE_WIDTH, Config.TEXTURE_HEIGHT),
            (0, 0, 0, 255)
        )
        
        # 이미지를 중앙에 배치
        offset_x = (Config.TEXTURE_WIDTH - img.width) // 2
        offset_y = (Config.TEXTURE_HEIGHT - img.height) // 2
        canvas.paste(img, (offset_x, offset_y))
        
        return canvas

    def _update_texture(self, data: list):
        """텍스처 생성 또는 업데이트"""
        if not self.texture_created:
            self._create_texture(data)
        else:
            dpg.set_value(self.texture_tag, data)

    def _create_texture(self, data: list):
        """새 텍스처 생성"""
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=Config.TEXTURE_WIDTH,
                height=Config.TEXTURE_HEIGHT,
                default_value=data,
                tag=self.texture_tag,
            )
        dpg.add_image(
            self.texture_tag,
            parent="image_area",
            tag=self.image_widget_tag,
        )
        self.texture_created = True

    def _display_scores(self, filename: str):
        """CSV 데이터에서 점수 로드 및 표시"""
        base_name = filename.replace(Config.IMAGE_PREFIX, "")
        row = self.df[self.df["file_name"] == base_name]

        if row.empty:
            self._show_no_data_message()
            return

        self._show_emotion_scores(row)

    def _show_no_data_message(self):
        """데이터 없음 메시지 표시"""
        dpg.set_value("score_text", "No row in CSV (face not detected).")
        dpg.set_value("match_text", "Face not detected")
        dpg.configure_item("match_text", color=[255, 0, 0])

    def _show_emotion_scores(self, row: pd.Series):
        """감정 점수 및 매치 결과 표시"""
        scores = row.iloc[0][Config.EMOTIONS].values
        max_idx = int(np.argmax(scores))
        max_emotion = Config.EMOTIONS[max_idx]
        max_score = scores[max_idx]

        # ✅ 감정과 점수를 페어로 만들고 점수 높은 순으로 정렬
        emotion_scores = list(zip(Config.EMOTIONS, scores))
        emotion_scores.sort(key=lambda x: x[1], reverse=True)
        
        # ✅ 점수만 표시 (타이틀은 별도)
        lines = [f"{e}: {s:.6f}" for e, s in emotion_scores]
        dpg.set_value("score_text", "\n".join(lines))

        # 매치 결과 표시
        is_match = max_emotion == self.selected_emotion
        self._show_match_result(max_emotion, max_score, is_match)
        
        # match인 경우 자동 저장
        if is_match:
            self._auto_save_match_result(row)

    def _show_match_result(self, max_emotion: str, max_score: float, is_match: bool = None):
        """매치 결과 표시"""
        if is_match is None:
            is_match = max_emotion == self.selected_emotion
            
        result_text = "MATCH" if is_match else "MISMATCH"
        color = [0, 255, 0] if is_match else [255, 0, 0]

        dpg.set_value(
            "match_text",
            f"{result_text}\n"
            f"max: {max_emotion} ({max_score})\n"
            f"vs target: {self.selected_emotion}",
        )
        dpg.configure_item("match_text", color=color)

    def _auto_save_match_result(self, row: pd.Series):
        """match인 경우 자동으로 'O' 저장"""
        if self.df is None:
            return
            
        idx = row.index[0]
        
        # 이미 저장된 경우 스킵
        if str(self.df.at[idx, "cmp_result"]) == "O":
            return
            
        # 'O' 저장
        self.df.at[idx, "cmp_result"] = "O"
        
        # CSV 저장
        self._save_csv()
        print(f"Auto-saved MATCH result for index {idx}")

    def update_progress(self):
        """진행도 업데이트"""
        total = max(1, len(self.image_files))
        current = self.current_idx + 1
        percentage = (current / total) * 100
        
        self._update_ui_message(
            "progress_text",
            f"Progress: {current}/{total} ({percentage:.1f}%)"
        )

    # ==================== USER INTERACTION ====================
    def on_user_judgment(self, sender, app_data, user_data):
        """사용자 감정 판별"""
        self.user_judgment = user_data
        dpg.set_value("judgment_status", f"Judged as: {user_data}")

    def on_emotion_change(self, sender, app_data, user_data):
        """타겟 감정 변경"""
        self.selected_emotion = app_data
        
        # ✅ 두 개의 콤보박스가 있으므로 둘 다 동기화
        if dpg.does_item_exist("target_emotion_combo"):
            dpg.set_value("target_emotion_combo", app_data)
        if dpg.does_item_exist("target_emotion_combo_large"):
            dpg.set_value("target_emotion_combo_large", app_data)
            
        if self.image_files:
            self.load_image()

    # ==================== NAVIGATION ====================
    def next_image(self, sender=None, app_data=None, user_data=None):
        """다음 이미지로 이동"""
        if not self._validate_navigation():
            return

        self._navigate_to_image(self.current_idx + 1, "next")

    def prev_image(self, sender=None, app_data=None, user_data=None):
        """이전 이미지로 이동"""
        if not self.image_files:
            return

        if self.current_idx > 0:
            self._navigate_to_image(self.current_idx - 1, "prev")

    def _navigate_to_image(self, new_idx: int, direction: str):
        """지정된 인덱스로 이동"""
        if 0 <= new_idx < len(self.image_files):
            self.current_idx = new_idx
            print(f"{direction.capitalize()}: Moving to index {self.current_idx}")
            
            try:
                self.load_image()
            except Exception as e:
                self._log_error("navigate", e)
        elif new_idx >= len(self.image_files):
            self._update_ui_message("image_text", "All images completed!")

    def _validate_navigation(self) -> bool:
        """네비게이션 가능 여부 확인"""
        return bool(self.image_files and self.df is not None)

    def save_current_and_state(self, sender=None, app_data=None, user_data=None):
        """현재 결과 저장 + 상태 저장"""
        try:
            self._save_current_result()
            self.save_state()
            dpg.set_value("judgment_status", "Saved!")
            print("Result and state saved successfully")
        except Exception as e:
            self._log_error("save", e)
            dpg.set_value("judgment_status", f"Save error: {e}")

    def _save_current_result(self):
        """현재 이미지 결과 CSV 저장 + mismatch 이동"""
        if not self._can_save_result():
            return

        filename = self.image_files[self.current_idx]
        base_name = filename.replace(Config.IMAGE_PREFIX, "")
        row = self.df[self.df["file_name"] == base_name]

        if row.empty:
            return

        idx = row.index[0]
        result = self._calculate_match_result(idx)
        
        # 결과 저장
        self._save_to_dataframe(idx, result)
        
        # ✅ MISMATCH 처리 (사용자 판별이 target과 같으면 이동하지 않음)
        if result == "x":
            should_move = self._should_move_to_mismatch()
            if should_move:
                self._handle_mismatch(filename)

        # CSV 저장
        self._save_csv()

    def _should_move_to_mismatch(self) -> bool:
        """mismatch 폴더로 이동할지 판단"""
        # 사용자 판별이 없으면 이동
        if not self.user_judgment:
            return True
        
        # 사용자 판별이 target emotion과 같으면 이동하지 않음
        if self.user_judgment == self.selected_emotion:
            print(f"User judgment matches target emotion ({self.selected_emotion}), not moving to mismatch")
            return False
        
        # 그 외의 경우는 이동
        return True

    def _can_save_result(self) -> bool:
        """저장 가능 여부 확인"""
        return bool(self.image_files and self.df is not None)

    def _calculate_match_result(self, idx: int) -> str:
        """매치 결과 계산"""
        scores = self.df.loc[idx, Config.EMOTIONS].values
        max_emotion = Config.EMOTIONS[int(np.argmax(scores))]
        return "O" if max_emotion == self.selected_emotion else "x"

    def _save_to_dataframe(self, idx: int, result: str):
        """DataFrame에 결과 저장"""
        self.df.at[idx, "cmp_result"] = str(result)
        
        if result == "x" and self.user_judgment:
            self.df.at[idx, "emotion"] = str(self.user_judgment)

    def _handle_mismatch(self, filename: str):
        """MISMATCH 이미지 처리"""
        self._move_to_mismatch(filename)

    def _move_to_mismatch(self, filename: str):
        """이미지를 mismatch 폴더로 이동"""
        if not self.image_folder:
            return
            
        src_path = Path(self.image_folder) / filename
        dst_path = Path(self.image_folder) / Config.MISMATCH_FOLDER / filename

        if src_path.exists():
            try:
                shutil.move(str(src_path), str(dst_path))
                print(f"Moved to mismatch: {filename}")
            except Exception as e:
                print(f"Error moving file: {e}")

    def _save_csv(self):
        """CSV 파일 저장"""
        try:
            self.df.to_csv(self.csv_file, index=False, na_rep="")
            print(f"CSV saved: {self.csv_file}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    # ==================== STATE MANAGEMENT ====================
    def save_state(self, sender=None, app_data=None, user_data=None):
        """현재 상태 저장"""
        state = {
            "current_idx": self.current_idx,
            "image_folder": self.image_folder,
            "csv_file": self.csv_file,
            "selected_emotion": self.selected_emotion,
        }
        try:
            with open(Config.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")

    def load_state(self):
        """저장된 상태 복원"""
        try:
            with open(Config.STATE_FILE, "r") as f:
                state = json.load(f)
            self._restore_state_from_dict(state)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading state: {e}")

    def _restore_state_from_dict(self, state: dict):
        """딕셔너리에서 상태 복원"""
        self.image_folder = state.get("image_folder")
        self.csv_file = state.get("csv_file")
        self.selected_emotion = state.get("selected_emotion", "neutral")
        self.current_idx = state.get("current_idx", 0)

        # ✅ UI 업데이트 (두 콤보박스 모두)
        if dpg.does_item_exist("target_emotion_combo"):
            dpg.set_value("target_emotion_combo", self.selected_emotion)
        if dpg.does_item_exist("target_emotion_combo_large"):
            dpg.set_value("target_emotion_combo_large", self.selected_emotion)

        # 데이터 복원
        if self.image_folder and self.csv_file:
            self._restore_data()

    def _restore_data(self):
        """이미지 파일 목록 및 CSV 복원"""
        if self.image_folder and os.path.exists(self.image_folder):
            self._load_image_files()
            self._ensure_mismatch_folder()

        if self.csv_file and os.path.exists(self.csv_file):
            try:
                self._load_csv_file(self.csv_file)
                self._update_restored_ui()
                
                if self.image_files and self.current_idx < len(self.image_files):
                    self.load_image()
            except Exception as e:
                print(f"Error restoring CSV: {e}")

    def _update_restored_ui(self):
        """복원 후 UI 업데이트"""
        self._update_ui_message(
            "image_text",
            f"Restored: {len(self.image_files)} images"
        )
        # ✅ 타이틀은 그대로 두고 본문만 업데이트
        self._update_ui_message(
            "score_text",
            f"CSV restored: {len(self.df)} rows\n"
            f"Progress: {self.current_idx}/{len(self.image_files)}"
        )

    # ==================== UTILITY METHODS ====================
    def _update_ui_message(self, tag: str, message: str):
        """UI 메시지 업데이트"""
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, message)

    def _show_error(self, tag: str, message: str):
        """에러 메시지 표시"""
        self._update_ui_message(tag, message)

    def _log_error(self, context: str, error: Exception):
        """에러 로깅"""
        print(f"Error in {context}: {error}")
        traceback.print_exc()


# ==================== MAIN ====================
def main():
    """메인 함수"""
    dpg.create_context()

    validator = EmotionValidator()

    # 키보드 단축키 등록
    with dpg.handler_registry():
        dpg.add_key_press_handler(
            key=dpg.mvKey_Left,
            callback=lambda s, a: validator.prev_image()
        )
        dpg.add_key_press_handler(
            key=dpg.mvKey_Right,
            callback=lambda s, a: validator.next_image()
        )

    # 뷰포트 설정
    dpg.create_viewport(title="Emotion Validation Tool", width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    
    # GUI 실행
    dpg.start_dearpygui()
    
    # 정리
    dpg.destroy_context()


if __name__ == "__main__":
    main()
