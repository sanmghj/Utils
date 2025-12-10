# emo_img_check.py
"""
얼굴 감정 분석 결과 검증 도구
[require install] pip install dearpygui pandas numpy pillow
"""

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
import os
import json
import shutil
from pathlib import Path
from typing import Optional, List
from PIL import Image


class Config:
    """설정 상수"""
    TEXTURE_SIZE = (500, 500)
    STATE_FILE = "validator_state.json"
    MISMATCH_FOLDER = "mismatch"
    IMAGE_PREFIX = "cropped_"
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
    EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


class EmotionValidator:
    """감정 분류 결과 검증 도구"""

    def __init__(self):
        self.csv_file: Optional[str] = None
        self.image_folder: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.image_files: List[str] = []
        self.current_idx: int = 0
        self.current_filename: Optional[str] = None
        self.selected_emotion: str = "neutral"
        self.user_judgment: Optional[str] = None
        self.texture_created = False

        self.setup_ui()
        self.load_state()

    # ==================== UI SETUP ====================
    def setup_ui(self):
        """메인 UI 구성"""
        with dpg.window(label="Emotion Validation Tool", width=1200, height=800, tag="main_window"):
            with dpg.group(horizontal=True):
                self._create_image_area()
                self._create_result_area()

    def _create_image_area(self):
        """이미지 영역"""
        with dpg.child_window(width=550, height=650, tag="image_area"):
            dpg.add_text("Select image folder and CSV file, then click Start.", tag="image_text")

    def _create_result_area(self):
        """결과 영역"""
        with dpg.child_window(width=600, height=650, tag="result_area"):
            dpg.add_text("Emotion Scores:", color=[255, 255, 0])
            dpg.add_text("waiting...", tag="score_text", wrap=580)
            dpg.add_text("Result: -", tag="match_text", color=[255, 255, 255])
            
            # ✅ (1) CSV 결과 표시 영역 추가
            dpg.add_text("", tag="csv_result_text", color=[200, 200, 200])
            
            dpg.add_text("Progress: 0/0 (0%)", tag="progress_text")
            
            dpg.add_separator()
            self._create_target_selector()
            
            dpg.add_separator()
            self._create_judgment_buttons()
            
            dpg.add_separator()
            self._create_nav_buttons()
            
            dpg.add_separator()
            self._create_file_controls()

    def _create_target_selector(self):
        """타겟 감정 선택"""
        dpg.add_text("Target emotion:", color=[255, 200, 0])
        dpg.add_combo(
            items=Config.EMOTIONS,
            default_value=self.selected_emotion,
            callback=self.on_emotion_change,
            tag="target_emotion_combo",
            width=250
        )

    def _create_judgment_buttons(self):
        """사용자 판별 버튼 (4x2 행렬)"""
        with dpg.group(horizontal=True):
            dpg.add_text("User Judgment (for mismatches):", color=[255, 255, 0])
            dpg.add_text("", tag="judgment_status", color=[0, 255, 0])
        
        # 4개씩 2줄
        for row_start in [0, 4]:
            with dpg.group(horizontal=True):
                emotions = Config.EMOTIONS[row_start:row_start+4] if row_start == 0 else Config.EMOTIONS[4:]
                for emotion in emotions:
                    dpg.add_button(
                        label=emotion.title(),
                        callback=self.on_user_judgment,
                        user_data=emotion,
                        width=130,
                        height=40
                    )
                
                # 두 번째 줄 마지막에 Save State
                if row_start == 4:
                    save_btn = dpg.add_button(
                        label="Save State",
                        width=130,
                        height=40,
                        callback=self.save_current_and_state
                    )
                    self._apply_save_button_theme(save_btn)

    def _apply_save_button_theme(self, btn):
        """Save 버튼 테마"""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 100, 150, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 120, 170, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (40, 90, 140, 255))
        dpg.bind_item_theme(btn, theme)

    def _create_nav_buttons(self):
        """네비게이션 버튼"""
        with dpg.group(horizontal=True):
            dpg.add_button(label="Prev", width=280, height=50, callback=self.prev_image)
            dpg.add_button(label="Next", width=280, height=50, callback=self.next_image)

    def _create_file_controls(self):
        """파일 선택 버튼"""
        dpg.add_text("File Controls:", color=[255, 255, 0])
        with dpg.group(horizontal=True):
            dpg.add_button(label="Image Folder", width=180, height=40, callback=self.select_image_folder)
            dpg.add_button(label="CSV File", width=180, height=40, callback=self.select_csv)
        dpg.add_button(label="Start Analysis", width=370, height=40, callback=self.start_analysis)

    # ==================== FILE DIALOGS ====================
    def select_image_folder(self, *args):
        """이미지 폴더 선택"""
        self._show_dialog("folder_dialog", True, self.on_folder_select)

    def select_csv(self, *args):
        """CSV 선택"""
        self._show_dialog("csv_dialog", False, self.on_csv_select, [(".csv", (0, 255, 0, 255))])

    def _show_dialog(self, tag, is_dir, callback, extensions=None):
        """다이얼로그 표시 헬퍼"""
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

        with dpg.file_dialog(
            modal=True,
            directory_selector=is_dir,
            show=False,
            callback=callback,
            tag=tag,
            width=700,
            height=400,
            default_path=self.image_folder or os.path.expanduser("~")
        ):
            if extensions:
                for ext, color in extensions:
                    dpg.add_file_extension(ext, color=color)
            dpg.add_file_extension("", color=(150, 150, 150, 255))

        dpg.show_item(tag)

    def on_folder_select(self, sender, app_data):
        """폴더 선택 후 처리"""
        folder = app_data.get("file_path_name")
        if not folder:
            return

        self.image_folder = folder
        self._load_image_files()
        Path(self.image_folder, Config.MISMATCH_FOLDER).mkdir(exist_ok=True)
        self._set_text("image_text", f"Folder loaded: {len(self.image_files)} images\n{folder}")

    def _load_image_files(self):
        """✅ 이미지 파일 목록 로드 (현재 폴더 + mismatch 폴더)"""
        if not self.image_folder:
            return
        
        # 현재 폴더의 이미지
        current_files = [
            f for f in os.listdir(self.image_folder)
            if f.startswith(Config.IMAGE_PREFIX) and f.lower().endswith(Config.IMAGE_EXTENSIONS)
        ]
        
        # mismatch 폴더의 이미지
        mismatch_path = Path(self.image_folder, Config.MISMATCH_FOLDER)
        mismatch_files = []
        if mismatch_path.exists():
            mismatch_files = [
                f for f in os.listdir(mismatch_path)
                if f.startswith(Config.IMAGE_PREFIX) and f.lower().endswith(Config.IMAGE_EXTENSIONS)
            ]
        
        # 두 목록 합치고 정렬
        all_files = current_files + mismatch_files
        self.image_files = sorted(list(set(all_files)))  # 중복 제거 후 정렬
        
        print(f"Loaded {len(current_files)} files from current, {len(mismatch_files)} from mismatch")

    def on_csv_select(self, sender, app_data):
        """CSV 선택 후 처리"""
        path = app_data.get("file_path_name")
        if not path or not path.lower().endswith(".csv"):
            self._set_text("score_text", "Error: Select a valid CSV file")
            return

        try:
            self.csv_file = path
            self.df = pd.read_csv(path)
            self._prepare_dataframe()
            self._set_text("score_text", f"CSV loaded: {len(self.df)} rows\n{os.path.basename(path)}")
        except Exception as e:
            self._set_text("score_text", f"Error loading CSV: {e}")
            self.csv_file = None
            self.df = None

    def _prepare_dataframe(self):
        """DataFrame 컬럼 준비"""
        for col in ["cmp_result", "emotion"]:
            if col not in self.df.columns:
                self.df[col] = ""
        
        result_cols = ["cmp_result", "emotion"]
        other_cols = [c for c in self.df.columns if c not in result_cols]
        self.df = self.df[other_cols + result_cols]
        self.df[result_cols] = self.df[result_cols].fillna("").astype(str)

    # ==================== MAIN FLOW ====================
    def start_analysis(self, *args):
        """분석 시작"""
        if not all([self.image_folder, self.image_files, self.csv_file, self.df is not None]):
            self._set_text("image_text", "Error: select image folder and CSV file first.")
            return
        self.load_image()

    def load_image(self):
        """✅ 이미지 로드 (현재 폴더 또는 mismatch 폴더에서)"""
        if not self.image_files or self.current_idx >= len(self.image_files):
            self._set_text("image_text", "All images processed.")
            return

        filename = self.image_files[self.current_idx]
        self.current_filename = filename
        
        # ✅ 현재 폴더에서 먼저 찾기
        img_path = Path(self.image_folder, filename)
        
        # ✅ 없으면 mismatch 폴더에서 찾기
        if not img_path.exists():
            img_path = Path(self.image_folder, Config.MISMATCH_FOLDER, filename)
        
        # ✅ 그래도 없으면 에러
        if not img_path.exists():
            self._set_text("image_text", f"File not found: {filename}")
            return
        
        self.user_judgment = None
        self._set_text("judgment_status", "")

        try:
            # ✅ 파일 위치 표시 (현재 폴더 or mismatch)
            location = "mismatch" if img_path.parent.name == Config.MISMATCH_FOLDER else "current"
            self._display_image(str(img_path), f"{filename} [{location}]")
            self._display_scores(filename)
        except Exception as e:
            print(f"Error loading image: {e}")
            self._set_text("image_text", f"Failed to load: {filename}")

        self._update_progress()

    def _display_image(self, img_path: str, filename: str):
        """이미지 표시"""
        img = Image.open(img_path).convert("RGBA")
        img.thumbnail(Config.TEXTURE_SIZE, Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", Config.TEXTURE_SIZE, (0, 0, 0, 255))
        offset = ((Config.TEXTURE_SIZE[0] - img.width) // 2, (Config.TEXTURE_SIZE[1] - img.height) // 2)
        canvas.paste(img, offset)

        data = (np.array(canvas, dtype=np.float32) / 255.0).flatten().tolist()

        if not self.texture_created:
            with dpg.texture_registry(show=False):
                dpg.add_dynamic_texture(*Config.TEXTURE_SIZE, default_value=data, tag="image_texture")
            dpg.add_image("image_texture", parent="image_area", tag="image_display")
            self.texture_created = True
        else:
            dpg.set_value("image_texture", data)

        self._set_text("image_text", filename)

    def _display_scores(self, filename: str):
        """점수 표시"""
        base_name = filename.replace(Config.IMAGE_PREFIX, "")
        row = self.df[self.df["file_name"] == base_name]

        if row.empty:
            self._set_text("score_text", "No row in CSV (face not detected).")
            self._set_text("match_text", "Face not detected", [255, 0, 0])
            # ✅ (1)(2) CSV 결과 표시 - 빈칸
            self._set_text("csv_result_text", "")
            return

        scores = row.iloc[0][Config.EMOTIONS].values
        max_idx = int(np.argmax(scores))
        max_emotion = Config.EMOTIONS[max_idx]
        max_score = scores[max_idx]

        # 점수 정렬 표시
        sorted_scores = sorted(zip(Config.EMOTIONS, scores), key=lambda x: x[1], reverse=True)
        self._set_text("score_text", "\n".join([f"{e}: {s:.6f}" for e, s in sorted_scores]))

        # 매치 결과
        is_match = max_emotion == self.selected_emotion
        result_text = "MATCH" if is_match else "MISMATCH"
        color = [0, 255, 0] if is_match else [255, 0, 0]
        self._set_text(
            "match_text",
            f"{result_text}\nmax: {max_emotion} ({max_score:.6f})\nvs target: {self.selected_emotion}",
            color
        )

        # ✅ (1)(2) CSV 결과 표시
        idx = row.index[0]
        cmp_result = str(self.df.at[idx, "cmp_result"]).strip()
        emotion = str(self.df.at[idx, "emotion"]).strip()
        
        # 빈 값 처리
        cmp_result_display = cmp_result if cmp_result and cmp_result != "nan" else "(empty)"
        emotion_display = emotion if emotion and emotion != "nan" else "(empty)"
        
        csv_info = f"CSV Result: {cmp_result_display}\nCSV Emotion: {emotion_display}"
        self._set_text("csv_result_text", csv_info, [200, 200, 200])

        # 자동 저장
        if is_match:
            if str(self.df.at[idx, "cmp_result"]) != "O":
                self.df.at[idx, "cmp_result"] = "O"
                self._save_csv()
                # 저장 후 다시 표시
                self._set_text("csv_result_text", f"CSV Result: O\nCSV Emotion: {emotion_display}", [200, 200, 200])

    def _update_progress(self):
        """✅ 진행도 업데이트 (CSV 행 기준)"""
        if self.df is None or not self.image_files:
            self._set_text("progress_text", "Progress: 0/0 (0%)")
            return
        
        # 현재 파일의 CSV 행 찾기
        filename = self.image_files[self.current_idx]
        base_name = filename.replace(Config.IMAGE_PREFIX, "")
        row = self.df[self.df["file_name"] == base_name]
        
        total_rows = len(self.df)  # 전체 행 수 (헤더 제외)
        
        if row.empty:
            # 현재 파일이 CSV에 없으면 대략적인 추정
            current_row = self.current_idx + 1
        else:
            # CSV에서 현재 행의 실제 인덱스 (0부터 시작이므로 +1)
            current_row = row.index[0] + 1
        
        percentage = (current_row / total_rows) * 100 if total_rows > 0 else 0
        
        self._set_text(
            "progress_text",
            f"Progress: {current_row}/{total_rows} ({percentage:.1f}%)"
        )

    # ==================== USER INTERACTION ====================
    def on_user_judgment(self, sender, app_data, user_data):
        """사용자 판별"""
        self.user_judgment = user_data
        self._set_text("judgment_status", f"Judged as: {user_data}")

    def on_emotion_change(self, sender, app_data, user_data):
        """타겟 감정 변경"""
        self.selected_emotion = app_data
        if self.image_files:
            self.load_image()

    # ==================== NAVIGATION ====================
    def next_image(self, *args):
        """다음 이미지로 이동 (저장/이동 없음)"""
        if not self.image_files or self.df is None:
            return
        
        old_filename = self.current_filename
        # ✅ 파일 목록 갱신 (현재 + mismatch)
        self._load_image_files()
        
        # 현재 파일이 목록에서 사라졌는지 확인
        if old_filename and old_filename not in self.image_files:
            # 파일이 삭제됨 -> 같은 인덱스 유지
            print(f"File {old_filename} removed, staying at index {self.current_idx}")
        else:
            # 파일이 여전히 있음 -> 다음으로 이동
            if self.current_idx < len(self.image_files) - 1:
                self.current_idx += 1
        
        # 이미지 로드
        if self.current_idx < len(self.image_files):
            self.load_image()
        else:
            self._set_text("image_text", "All images completed!")

    def prev_image(self, *args):
        """이전 이미지로 이동 (저장/이동 없음)"""
        if not self.image_files:
            return
        
        # ✅ 파일 목록 갱신 (현재 + mismatch)
        self._load_image_files()
        
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def save_current_and_state(self, *args):
        """✅ Save State 버튼: 결과 저장 + 파일 이동 + 상태 저장"""
        try:
            if not self.image_files or self.df is None:
                self._set_text("judgment_status", "No data to save")
                return
            
            filename = self.image_files[self.current_idx]
            
            # ✅ 현재 파일이 어디에 있는지 확인
            current_path = Path(self.image_folder, filename)
            mismatch_path = Path(self.image_folder, Config.MISMATCH_FOLDER, filename)
            is_in_mismatch = mismatch_path.exists() and not current_path.exists()
            
            if is_in_mismatch:
                # ✅ mismatch 폴더에 있는 경우 재분석 로직
                self._handle_mismatch_reanalysis()
            else:
                # ✅ current 폴더에 있는 경우 기존 로직
                should_move = self._should_move_to_mismatch()
                
                if should_move:
                    self._save_current_result_and_move()
                    self._set_text("judgment_status", "Saved & Moved to mismatch!")
                else:
                    self._save_current_result_without_move()
                    self._set_text("judgment_status", "Saved!")
            
            # 상태 저장
            self.save_state()
            
            # 진행도 업데이트
            self._update_progress()
            
        except Exception as e:
            self._set_text("judgment_status", f"Save error: {e}")

    def _handle_mismatch_reanalysis(self):
        """✅ mismatch 폴더의 이미지 재분석 처리"""
        if not self.image_files or self.df is None:
            return

        filename = self.image_files[self.current_idx]
        base_name = filename.replace(Config.IMAGE_PREFIX, "")
        row = self.df[self.df["file_name"] == base_name]

        if row.empty:
            return

        idx = row.index[0]
        scores = self.df.loc[idx, Config.EMOTIONS].values
        max_emotion = Config.EMOTIONS[int(np.argmax(scores))]
        
        # 파일 경로
        mismatch_path = Path(self.image_folder, Config.MISMATCH_FOLDER, filename)
        current_path = Path(self.image_folder, filename)
        
        # user judgment가 없으면 현재 상태 유지
        if not self.user_judgment:
            self._set_text("judgment_status", "Please select user judgment first")
            return
        
        # (2) user judgment가 target emotion과 동일 -> current 폴더로 이동
        if self.user_judgment == self.selected_emotion:
            result = "O"
            self.df.at[idx, "cmp_result"] = str(result)
            self.df.at[idx, "emotion"] = str(self.user_judgment)
            self._save_csv()
            
            # current 폴더로 이동
            if mismatch_path.exists():
                shutil.move(str(mismatch_path), str(current_path))
                print(f"Moved back to current: {filename}")
                self._set_text("judgment_status", "Saved & Moved to current!")
            
            # CSV 결과 표시 업데이트
            self._set_text("csv_result_text", f"CSV Result: {result}\nCSV Emotion: {self.user_judgment}", [200, 200, 200])
        
        # (1) user judgment가 target emotion과 다름 -> mismatch 폴더 유지
        else:
            result = "x"
            self.df.at[idx, "cmp_result"] = str(result)
            self.df.at[idx, "emotion"] = str(self.user_judgment)
            self._save_csv()
            
            print(f"Kept in mismatch: {filename}")
            self._set_text("judgment_status", "Saved (kept in mismatch)")
            
            # CSV 결과 표시 업데이트
            self._set_text("csv_result_text", f"CSV Result: {result}\nCSV Emotion: {self.user_judgment}", [200, 200, 200])

    def _should_move_to_mismatch(self) -> bool:
        """mismatch 폴더로 이동할지 판단"""
        if not self.image_files or self.df is None:
            return False
        
        filename = self.image_files[self.current_idx]
        base_name = filename.replace(Config.IMAGE_PREFIX, "")
        row = self.df[self.df["file_name"] == base_name]
        
        if row.empty:
            return False
        
        # CSV에서 최대 감정 찾기
        scores = row.iloc[0][Config.EMOTIONS].values
        max_emotion = Config.EMOTIONS[int(np.argmax(scores))]
        
        # (2-1) csv결과와 target emotion이 같은 경우 -> 유지
        if max_emotion == self.selected_emotion:
            return False
        
        # (2-2) csv결과와 target emotion이 다르지만 user judgment가 없는 경우 -> 유지
        if not self.user_judgment:
            return False
        
        # (2-3) csv결과와 target emotion이 다르면서 user judgment와 target emotion이 같은 경우 -> 유지
        if self.user_judgment == self.selected_emotion:
            return False
        
        # (2-4) csv결과와 target emotion이 다르면서 user judgment와 target emotion이 다른 경우 -> 이동
        return True

    def _save_current_result_and_move(self):
        """✅ 현재 결과 저장 후 mismatch로 이동"""
        if not self.image_files or self.df is None:
            return

        filename = self.image_files[self.current_idx]
        base_name = filename.replace(Config.IMAGE_PREFIX, "")
        row = self.df[self.df["file_name"] == base_name]

        if row.empty:
            return

        idx = row.index[0]
        scores = self.df.loc[idx, Config.EMOTIONS].values
        max_emotion = Config.EMOTIONS[int(np.argmax(scores))]
        result = "O" if max_emotion == self.selected_emotion else "x"

        # 결과 저장
        self.df.at[idx, "cmp_result"] = str(result)
        
        if self.user_judgment:
            self.df.at[idx, "emotion"] = str(self.user_judgment)

        self._save_csv()
        
        # CSV 결과 표시 업데이트
        emotion_display = str(self.user_judgment) if self.user_judgment else "(empty)"
        self._set_text("csv_result_text", f"CSV Result: {result}\nCSV Emotion: {emotion_display}", [200, 200, 200])
        
        # ✅ mismatch로 이동 (현재 폴더에 있는 경우만)
        src = Path(self.image_folder, filename)
        dst = Path(self.image_folder, Config.MISMATCH_FOLDER, filename)
        
        if src.exists():
            # 현재 폴더에 있으면 mismatch로 이동
            shutil.move(str(src), str(dst))
            print(f"Moved to mismatch: {filename}")
        else:
            # 이미 mismatch에 있으면 이동하지 않음
            print(f"Already in mismatch: {filename}")

    def _save_current_result_without_move(self):
        """현재 결과 저장 (이동 안 함)"""
        if not self.image_files or self.df is None:
            return

        filename = self.image_files[self.current_idx]
        base_name = filename.replace(Config.IMAGE_PREFIX, "")
        row = self.df[self.df["file_name"] == base_name]

        if row.empty:
            return

        idx = row.index[0]
        scores = self.df.loc[idx, Config.EMOTIONS].values
        max_emotion = Config.EMOTIONS[int(np.argmax(scores))]
        result = "O" if max_emotion == self.selected_emotion else "x"

        # 결과 저장
        self.df.at[idx, "cmp_result"] = str(result)
        
        if self.user_judgment:
            self.df.at[idx, "emotion"] = str(self.user_judgment)

        self._save_csv()
        
        # CSV 결과 표시 업데이트
        emotion_display = str(self.user_judgment) if self.user_judgment else "(empty)"
        self._set_text("csv_result_text", f"CSV Result: {result}\nCSV Emotion: {emotion_display}", [200, 200, 200])
        
        print(f"Saved result for: {filename} (kept in place)")

    def save_state(self, *args):
        """상태 저장"""
        state = {
            "current_filename": self.current_filename,
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
        """상태 복원"""
        try:
            with open(Config.STATE_FILE, "r") as f:
                state = json.load(f)
            
            self.image_folder = state.get("image_folder")
            self.csv_file = state.get("csv_file")
            self.selected_emotion = state.get("selected_emotion", "neutral")
            self.current_filename = state.get("current_filename")

            if dpg.does_item_exist("target_emotion_combo"):
                dpg.set_value("target_emotion_combo", self.selected_emotion)

            if self.image_folder and self.csv_file and os.path.exists(self.csv_file):
                self._load_image_files()
                
                # 파일 이름으로 인덱스 찾기
                if self.current_filename and self.current_filename in self.image_files:
                    self.current_idx = self.image_files.index(self.current_filename)
                    print(f"Restored to file: {self.current_filename} at index {self.current_idx}")
                else:
                    # 파일을 찾지 못한 경우 처음부터
                    self.current_idx = 0
                    self.current_filename = self.image_files[0] if self.image_files else None
                    print(f"File not found, starting from index 0")
                
                self.df = pd.read_csv(self.csv_file)
                self._prepare_dataframe()
                
                if self.image_files and self.current_idx < len(self.image_files):
                    self.load_image()
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading state: {e}")

    # ==================== UTILITY ====================
    def _set_text(self, tag, text, color=None):
        """텍스트 설정"""
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, text)
            if color:
                dpg.configure_item(tag, color=color)

    def _save_csv(self):
        """CSV 파일 저장"""
        if self.csv_file and self.df is not None:
            try:
                self.df.to_csv(self.csv_file, index=False)
                print(f"CSV saved: {self.csv_file}")
            except Exception as e:
                print(f"Error saving CSV: {e}")


# ==================== MAIN ====================
def main():
    dpg.create_context()
    validator = EmotionValidator()

    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_Left, callback=lambda s, a: validator.prev_image())
        dpg.add_key_press_handler(dpg.mvKey_Right, callback=lambda s, a: validator.next_image())
        # ✅ (1) 's' 또는 'S' 키로 Save State 실행
        dpg.add_key_press_handler(dpg.mvKey_S, callback=lambda s, a: validator.save_current_and_state())

    dpg.create_viewport(title="Emotion Validation Tool", width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
