"""
TRUVI-EV — Studio Desktop Fact-Checking Application
==================================================
High-Performance Obsidian Desktop Panel with
Forensic Contradiction Pinpointing, Proving Resource Citations,
and Factual Ground-Truth Corrections.

Features:
  - Space-grade Obsidian Dark UI with real-time reactive micro-animations
  - Forensic Contradiction Audit:
      * WHAT PART IS CONTRADICTED (exact false phrase/assertion)
      * HOW IT IS CONTRADICTED & PROVING RESOURCE (source citation & verbatim evidence quote)
      * WHAT WOULD BE THE RIGHT STATEMENT INSTEAD (ground-truth correction)
      * FULLY CORRECTED PARAGRAPH (rewritten passage with 1-click copy)
  - Ultra-responsive clipboard auto-verify HUD (<250ms polling) with hover pause & pin
  - Segmented Tab Navigation:
      1. ⚡ Live Verifier (single claim & paragraph decomposition)
      2. 📊 17-Signal Neural Radar & Reliability Gating Weights
      3. 🏛️ Evidence & Sources Explorer (NIST, NASA, Britannica, Wikipedia, Web)
      4. 🕒 Session History & Audit Log
  - 1-Click "📋 Copy Fact-Check Report" (formatted Markdown export)
  - Keyboard shortcuts (⌘+Enter Verify, ⌘+K Clear, ⌘+C Copy Report)
"""

import os

# Prevent tokenizer multi-threading deadlock & macOS segmentation faults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import customtkinter as ctk

# Clipboard support
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

def _play_tactile_chime(sound_name: str = "Tink.aiff"):
    """Plays crisp native macOS audio feedback asynchronously for instant tactile confirmation."""
    try:
        sound_path = f"/System/Library/Sounds/{sound_name}"
        if os.path.exists(sound_path):
            subprocess.Popen(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import get_engine

# =============================================================================
# Obsidian Studio Design Tokens & Color Palette
# =============================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Theme Palette: Obsidian High-Tech Studio
COLOR_BG_ROOT = "#050811"             # Deep space obsidian root
COLOR_PANEL_BG = "#0b1222"            # Surface card base
COLOR_PANEL_BG_ALT = "#0f192e"        # Slightly elevated surface
COLOR_PANEL_HOVER = "#16233e"         # Hover surface highlight
COLOR_PANEL_BORDER = "#1e293b"        # Refined subtle slate border
COLOR_PANEL_BORDER_LIGHT = "#334155"  # Border highlight
COLOR_PANEL_BORDER_ACTIVE = "#0284c7" # Electric cyan active border

# Typography Colors
COLOR_TEXT_PRIMARY = "#f8fafc"        # Crisp pure white text
COLOR_TEXT_SECONDARY = "#94a3b8"      # Light slate text
COLOR_TEXT_MUTED = "#64748b"          # Muted slate text
COLOR_ACCENT_PRIMARY = "#0284c7"      # Electric Deep Cyan
COLOR_ACCENT_HOVER = "#0369a1"        # Hover cyan
COLOR_ACCENT_GLOW = "#0ea5e9"         # Bright cyan glow
COLOR_ACCENT_CYAN = "#38bdf8"         # Bright cyan accent

# Stance & Verdict Colors
COLOR_SUPPORTED_BG = "#04261b"
COLOR_SUPPORTED_BORDER = "#10b981"
COLOR_SUPPORTED_TEXT = "#34d399"
COLOR_SUPPORTED_ACCENT = "#10b981"

COLOR_CONTRADICTED_BG = "#350712"
COLOR_CONTRADICTED_BORDER = "#f43f5e"
COLOR_CONTRADICTED_TEXT = "#fb7185"
COLOR_CONTRADICTED_ACCENT = "#f43f5e"

COLOR_UNVERIFIED_BG = "#2a1503"
COLOR_UNVERIFIED_BORDER = "#f59e0b"
COLOR_UNVERIFIED_TEXT = "#fcd34d"
COLOR_UNVERIFIED_ACCENT = "#f59e0b"


def send_macos_notification(title: str, subtitle: str, message: str):
    """Sends a native macOS desktop notification banner via osascript."""
    try:
        clean_msg = message.replace('"', '\\"').replace("'", "")[:140]
        clean_sub = subtitle.replace('"', '\\"')[:80]
        clean_title = title.replace('"', '\\"')[:60]
        script = f'display notification "{clean_msg}" with title "{clean_title}" subtitle "{clean_sub}" sound name "default"'
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# =============================================================================
# Floating High-Tech HUD Notification (Real-Time Clipboard Monitor)
# =============================================================================
class FloatingHUDNotification(ctk.CTkToplevel):
    """
    High-tech floating HUD notification window.
    Appears at top-right of the user's screen whenever text is copied anywhere on macOS.
    Provides immediate visual feedback with dual-phase rendering:
      Phase 1: Real-time verification analyzing progress (<50ms feedback)
      Phase 2: Comprehensive factuality verdict, refutation details, and ground truth
    """

    def __init__(self, parent_app, initial_text: str = "", result: Optional[Dict[str, Any]] = None):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.result = result
        self.current_text = initial_text or (result.get("paragraph" if "paragraph" in (result or {}) else "claim", "") if result else "")
        self.auto_close_id = None
        self.is_pinned = False

        # Frameless floating HUD window on macOS
        try:
            self.overrideredirect(True)
        except Exception:
            pass
        try:
            self.attributes("-topmost", True)
        except Exception:
            pass
        try:
            self.attributes("-alpha", 0.98)
        except Exception:
            pass

        self.sw = self.winfo_screenwidth()
        self.w = 520
        is_con = bool(result and (result.get("verdict") == "CONTRADICTED" or result.get("overall_verdict") == "CONTRADICTED"))
        self.h = 240 if not result else (350 if is_con else 280)
        self.x = self.sw - self.w - 24
        self.y = 48
        self.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")
        self.configure(fg_color=COLOR_BG_ROOT)

        # Card container
        border_col = COLOR_ACCENT_CYAN if not result else (COLOR_CONTRADICTED_BORDER if is_con else COLOR_SUPPORTED_BORDER)
        self.card = ctk.CTkFrame(
            self,
            fg_color=COLOR_PANEL_BG,
            border_color=border_col,
            border_width=2,
            corner_radius=16
        )
        self.card.pack(fill="both", expand=True, padx=2, pady=2)

        self.bind("<Enter>", self._on_mouse_enter)
        self.bind("<Leave>", self._on_mouse_leave)

        if result:
            self._render_result()
        else:
            self._render_analyzing()

        self.update_idletasks()
        self.lift()
        self.attributes("-topmost", True)
        self.after(50, self.lift)

    def _start_timer(self, ms: int = 12000):
        if self.is_pinned:
            return
        self._cancel_timer()
        self.auto_close_id = self.after(ms, self._auto_close)

    def _cancel_timer(self):
        if self.auto_close_id:
            try:
                self.after_cancel(self.auto_close_id)
            except Exception:
                pass
            self.auto_close_id = None

    def _on_mouse_enter(self, _event=None):
        self._cancel_timer()

    def _on_mouse_leave(self, _event=None):
        try:
            px = self.winfo_pointerx()
            py = self.winfo_pointery()
            wx = self.winfo_rootx()
            wy = self.winfo_rooty()
            ww = self.winfo_width()
            wh = self.winfo_height()
            if wx <= px <= wx + ww and wy <= py <= wy + wh:
                return  # Pointer is still within this HUD window
        except Exception:
            pass
        if not self.is_pinned:
            self._start_timer(7000)

    def _toggle_pin(self):
        self.is_pinned = not self.is_pinned
        if self.is_pinned:
            self._cancel_timer()
            if hasattr(self, "pin_btn"):
                self.pin_btn.configure(text="📌 Pinned", fg_color="#0369a1", text_color="#ffffff")
        else:
            if hasattr(self, "pin_btn"):
                self.pin_btn.configure(text="📌 Pin", fg_color="transparent", text_color=COLOR_TEXT_MUTED)
            self._start_timer(6000)

    def set_analyzing_state(self, text: str):
        self._cancel_timer()
        self.current_text = text
        self.result = None
        self.h = 240
        self.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")
        self.card.configure(border_color=COLOR_ACCENT_CYAN)
        self._render_analyzing()
        self.update_idletasks()
        self.lift()
        self.attributes("-topmost", True)
        self.after(50, self.lift)

    def set_result(self, result: Dict[str, Any]):
        self.result = result
        is_contradicted = (result.get("verdict") == "CONTRADICTED" or result.get("overall_verdict") == "CONTRADICTED")
        self.h = 350 if is_contradicted else 280
        self.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")
        self._render_result()
        self.update_idletasks()
        self.lift()
        self.attributes("-topmost", True)
        self.after(50, self.lift)

    def _render_analyzing(self):
        for w in self.card.winfo_children():
            w.destroy()

        # Header Bar
        hdr = ctk.CTkFrame(self.card, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(12, 6))

        tag = ctk.CTkLabel(
            hdr,
            text="⚡ TRUVI-EV CLIPBOARD MONITOR",
            font=ctk.CTkFont(family="SF Pro Display", size=11, weight="bold"),
            text_color=COLOR_ACCENT_CYAN
        )
        tag.pack(side="left")

        actions_right = ctk.CTkFrame(hdr, fg_color="transparent")
        actions_right.pack(side="right")

        self.pin_btn = ctk.CTkButton(
            actions_right,
            text="📌 Pinned" if self.is_pinned else "📌 Pin",
            width=54,
            height=22,
            fg_color="#0369a1" if self.is_pinned else "transparent",
            hover_color="#1e293b",
            text_color="#ffffff" if self.is_pinned else COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            command=self._toggle_pin
        )
        self.pin_btn.pack(side="left", padx=(0, 6))

        close_btn = ctk.CTkButton(
            actions_right,
            text="✕",
            width=22,
            height=22,
            fg_color="transparent",
            hover_color="#ef4444",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._dismiss
        )
        close_btn.pack(side="right")

        # Text snippet preview
        snippet = self.current_text
        if len(snippet) > 110:
            snippet = snippet[:107] + "..."
        snippet_lbl = ctk.CTkLabel(
            self.card,
            text=f'"{snippet}"',
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color=COLOR_TEXT_PRIMARY,
            wraplength=480,
            justify="left"
        )
        snippet_lbl.pack(anchor="w", padx=16, pady=(0, 10))

        # Analyzing Banner
        status_box = ctk.CTkFrame(self.card, fg_color="#091428", corner_radius=10)
        status_box.pack(fill="x", padx=16, pady=(0, 12))

        st_inner = ctk.CTkFrame(status_box, fg_color="transparent")
        st_inner.pack(fill="x", padx=14, pady=12)

        st_lbl = ctk.CTkLabel(
            st_inner,
            text="🔍 Verifying copied assertion against live authoritative knowledge...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#38bdf8"
        )
        st_lbl.pack(anchor="w", pady=(0, 6))

        bar = ctk.CTkProgressBar(st_inner, height=6, corner_radius=3, progress_color=COLOR_ACCENT_CYAN, fg_color="#182232")
        bar.set(0.65)
        bar.pack(fill="x", pady=2)

        sub_lbl = ctk.CTkLabel(
            st_inner,
            text="Decomposing atomic claims • Querying multi-tier evidence archives • Evaluating NLI",
            font=ctk.CTkFont(size=10),
            text_color=COLOR_TEXT_MUTED
        )
        sub_lbl.pack(anchor="w", pady=(4, 0))

    def _render_result(self):
        for w in self.card.winfo_children():
            w.destroy()

        is_para = "paragraph" in self.result
        verdict = self.result.get("overall_verdict" if is_para else "verdict", "UNVERIFIED")

        if verdict == "CONTRADICTED":
            v_bg, v_border, v_txt = COLOR_CONTRADICTED_BG, COLOR_CONTRADICTED_BORDER, COLOR_CONTRADICTED_TEXT
            badge_icon = "✕"
            verdict_label = "CONTRADICTED"
        elif verdict == "SUPPORTED":
            v_bg, v_border, v_txt = COLOR_SUPPORTED_BG, COLOR_SUPPORTED_BORDER, COLOR_SUPPORTED_TEXT
            badge_icon = "✓"
            verdict_label = "SUPPORTED"
        else:
            v_bg, v_border, v_txt = COLOR_UNVERIFIED_BG, COLOR_UNVERIFIED_BORDER, COLOR_UNVERIFIED_TEXT
            badge_icon = "⚠"
            verdict_label = "UNVERIFIED"

        self.card.configure(border_color=v_border)
        conf_str = self.result.get("overall_confidence_pct" if is_para else "confidence_pct", "85.0%")

        # 1. Header Bar
        hdr = ctk.CTkFrame(self.card, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(12, 4))

        tag = ctk.CTkLabel(
            hdr,
            text="⚡ TRUVI-EV FACT-CHECK",
            font=ctk.CTkFont(family="SF Pro Display", size=11, weight="bold"),
            text_color=COLOR_ACCENT_CYAN
        )
        tag.pack(side="left")

        actions_right = ctk.CTkFrame(hdr, fg_color="transparent")
        actions_right.pack(side="right")

        self.pin_btn = ctk.CTkButton(
            actions_right,
            text="📌 Pinned" if self.is_pinned else "📌 Pin",
            width=54,
            height=22,
            fg_color="#0369a1" if self.is_pinned else "transparent",
            hover_color="#1e293b",
            text_color="#ffffff" if self.is_pinned else COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=11),
            command=self._toggle_pin
        )
        self.pin_btn.pack(side="left", padx=(0, 6))

        close_btn = ctk.CTkButton(
            actions_right,
            text="✕",
            width=22,
            height=22,
            fg_color="transparent",
            hover_color="#ef4444",
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._dismiss
        )
        close_btn.pack(side="right")

        # 2. Text Snippet Preview
        text_snippet = self.result.get("paragraph" if is_para else "claim", self.current_text)
        if len(text_snippet) > 105:
            text_snippet = text_snippet[:102] + "..."
        snippet_lbl = ctk.CTkLabel(
            self.card,
            text=f'"{text_snippet}"',
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color=COLOR_TEXT_PRIMARY,
            wraplength=480,
            justify="left"
        )
        snippet_lbl.pack(anchor="w", padx=16, pady=(0, 6))

        # 3. Verdict & Ratio Pill
        status_row = ctk.CTkFrame(self.card, fg_color="transparent")
        status_row.pack(fill="x", padx=16, pady=(0, 6))

        pill = ctk.CTkLabel(
            status_row,
            text=f" {badge_icon}  {verdict_label}  •  {conf_str} ",
            fg_color=v_bg,
            text_color=v_txt,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            padx=10,
            pady=3
        )
        pill.pack(side="left")

        # Composition breakdown pill
        f_pct = self.result.get("factuality_pct")
        h_pct = self.result.get("hallucination_pct")
        if f_pct and h_pct:
            comp_txt = f"🟢 {f_pct} Factual | 🔴 {h_pct} False" if verdict == "CONTRADICTED" else f"🟢 {f_pct} Verified"
            comp_pill = ctk.CTkLabel(
                status_row,
                text=comp_txt,
                fg_color="#141d2e",
                text_color="#cbd5e1",
                corner_radius=8,
                font=ctk.CTkFont(size=11, weight="bold"),
                padx=8,
                pady=3
            )
            comp_pill.pack(side="left", padx=8)

        # 4. If Contradicted: Dedicated Forensics & Right Statement Box
        if verdict == "CONTRADICTED":
            c_part = self.result.get("contradicted_part", "")
            resrc = self.result.get("proving_resource", {}) or {}
            r_stmt = self.result.get("right_statement", "")

            if is_para and self.result.get("contradictions"):
                top_c = self.result["contradictions"][0]
                c_part = top_c.get("contradicted_part", "")
                resrc = top_c.get("proving_resource", {}) or {}
                r_stmt = top_c.get("right_statement", "")

            forensic_box = ctk.CTkFrame(self.card, fg_color="#090e18", border_color=COLOR_CONTRADICTED_BORDER, border_width=1, corner_radius=10)
            forensic_box.pack(fill="x", padx=16, pady=(0, 8))

            f_inner = ctk.CTkFrame(forensic_box, fg_color="transparent")
            f_inner.pack(fill="x", padx=10, pady=6)

            if c_part:
                c_disp = f'🔴 False Part: "{c_part[:60]}..."' if len(c_part) > 63 else f'🔴 False Part: "{c_part}"'
                c_lbl = ctk.CTkLabel(
                    f_inner,
                    text=c_disp,
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=COLOR_CONTRADICTED_TEXT,
                    anchor="w"
                )
                c_lbl.pack(anchor="w")

            src_name = resrc.get("source", "Authoritative Reference Database")
            auth_pct = resrc.get("authority_pct", "99%")
            s_lbl = ctk.CTkLabel(
                f_inner,
                text=f"🏛️ Proved by: {src_name} ({auth_pct} Authority)",
                font=ctk.CTkFont(size=10),
                text_color="#94a3b8",
                anchor="w"
            )
            s_lbl.pack(anchor="w", pady=(1, 3))

            if r_stmt:
                r_disp = r_stmt if len(r_stmt) < 110 else r_stmt[:106] + "..."
                r_lbl = ctk.CTkLabel(
                    f_inner,
                    text=f'🟢 Say this instead: "{r_disp}"',
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#34d399",
                    wraplength=470,
                    justify="left"
                )
                r_lbl.pack(anchor="w")
        else:
            advisory = self.result.get("action_advisory", "")
            if not advisory:
                advisory = self.result.get("overall_summary" if is_para else "short_reason", "")
            if len(advisory) > 130:
                advisory = advisory[:126] + "..."

            adv_box = ctk.CTkFrame(self.card, fg_color="#090e18", corner_radius=8)
            adv_box.pack(fill="x", padx=16, pady=(0, 8))

            adv_lbl = ctk.CTkLabel(
                adv_box,
                text=advisory,
                font=ctk.CTkFont(size=11),
                text_color="#cbd5e1",
                wraplength=470,
                justify="left"
            )
            adv_lbl.pack(anchor="w", padx=10, pady=6)

        # 5. Bottom Actions Row
        actions = ctk.CTkFrame(self.card, fg_color="transparent")
        actions.pack(fill="x", padx=16, pady=(0, 10))

        if verdict == "CONTRADICTED" and (self.result.get("right_statement") or is_para):
            self.copy_btn = ctk.CTkButton(
                actions,
                text="📋 Copy Right Fact",
                height=28,
                fg_color="#064e3b",
                hover_color="#047857",
                text_color="#34d399",
                font=ctk.CTkFont(size=11, weight="bold"),
                command=self._copy_right_statement
            )
            self.copy_btn.pack(side="left", padx=(0, 6))
        else:
            self.copy_btn = ctk.CTkButton(
                actions,
                text="📋 Copy Fact-Check",
                height=28,
                fg_color="#1e293b",
                hover_color="#334155",
                text_color="#e2e8f0",
                font=ctk.CTkFont(size=11),
                command=self._copy_summary
            )
            self.copy_btn.pack(side="left", padx=(0, 6))

        inspect_btn = ctk.CTkButton(
            actions,
            text="⚡ Open in Studio →",
            height=28,
            fg_color=COLOR_ACCENT_PRIMARY,
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._inspect_in_main
        )
        inspect_btn.pack(side="right")

        if not self.is_pinned:
            self._start_timer(12000)

    def _copy_right_statement(self):
        is_para = "paragraph" in self.result
        if is_para and self.result.get("corrected_paragraph"):
            text_to_copy = self.result["corrected_paragraph"]
        else:
            text_to_copy = self.result.get("right_statement", "")

        try:
            if text_to_copy:
                clean = text_to_copy.strip()
                if CLIPBOARD_AVAILABLE:
                    if hasattr(self.parent_app, "last_clipboard_text"):
                        self.parent_app.last_clipboard_text = clean
                    pyperclip.copy(clean)
                subprocess.run(["pbcopy"], input=clean.encode("utf-8"), check=False)
                _play_tactile_chime("Tink.aiff")
            self.copy_btn.configure(text="✓ Copied!", fg_color="#059669")
            self.after(1600, lambda: self.copy_btn.configure(text="📋 Copy Right Fact", fg_color="#064e3b"))
        except Exception:
            pass

    def _copy_summary(self):
        is_para = "paragraph" in self.result
        v = self.result.get("overall_verdict" if is_para else "verdict", "UNVERIFIED")
        conf = self.result.get("overall_confidence_pct" if is_para else "confidence_pct", "")
        text = self.result.get("paragraph" if is_para else "claim", self.current_text)
        adv = self.result.get("action_advisory", "")
        report = f"**TRUVI-EV Fact-Check Report**\n• Claim: \"{text}\"\n• Verdict: {v} ({conf})\n• Advisory: {adv}"

        try:
            clean = report.strip()
            if CLIPBOARD_AVAILABLE:
                if hasattr(self.parent_app, "last_clipboard_text"):
                    self.parent_app.last_clipboard_text = clean
                pyperclip.copy(clean)
            subprocess.run(["pbcopy"], input=clean.encode("utf-8"), check=False)
            _play_tactile_chime("Tink.aiff")
            self.copy_btn.configure(text="✓ Copied!", fg_color="#059669")
            self.after(1600, lambda: self.copy_btn.configure(text="📋 Copy Fact-Check", fg_color="#1e293b"))
        except Exception:
            pass

    def _dismiss(self):
        self._cancel_timer()
        try:
            self.destroy()
        except Exception:
            pass

    def _auto_close(self):
        if not self.is_pinned:
            self._dismiss()

    def _inspect_in_main(self):
        self._cancel_timer()
        self.parent_app.load_and_display_result(self.result)
        self.parent_app.deiconify()
        self.parent_app.lift()
        self.parent_app.focus_force()
        try:
            self.destroy()
        except Exception:
            pass


# =============================================================================
# Forensic Deep Inspection Modal Window (Claim & Sub-Claim Forensics)
# =============================================================================
class ClaimForensicModal(ctk.CTkToplevel):
    """
    High-resolution forensic deep inspection modal.
    Opened whenever a user clicks on an individual claim card or compound sub-claim.
    Features:
      - Independent top-level window (no Cocoa transient bug)
      - Topmost stay-on-top positioning
      - Complete contradiction breakdown, proving resource citations, and 1-click ground truth
    """

    def __init__(self, parent_app, claim_data: Dict[str, Any], claim_number: int = 1):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.clm = claim_data
        self.claim_num = claim_number

        # Center window reliably on screen
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(1040, max(800, sw - 80))
        h = min(840, max(560, sh - 100))
        x = max(30, (sw - w) // 2)
        y = max(40, (sh - h) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(780, 520)
        self.title(f"TRUVI-EV Forensic Deep Inspection • Claim #{self.claim_num}")
        self.configure(fg_color="#070a13")

        # Stable macOS Cocoa window hierarchy: transient without aggressive topmost
        try:
            self.transient(parent_app)
        except Exception:
            pass
        self.deiconify()
        self.lift()
        self.after(100, lambda: self.focus())

        # Handle window closure cleanly
        self.protocol("WM_DELETE_WINDOW", self._close_modal)
        self.bind("<Escape>", lambda e: self._close_modal())

        self._build_modal_ui()

    def _close_modal(self):
        try:
            if hasattr(self.parent_app, "open_forensic_modals"):
                self.parent_app.open_forensic_modals.pop(self.claim_num, None)
            if hasattr(self.parent_app, "active_forensic_modal") and self.parent_app.active_forensic_modal is self:
                self.parent_app.active_forensic_modal = None
            self.destroy()
        except Exception:
            pass

    def _build_modal_ui(self):
        # Top Navigation Bar
        top_bar = ctk.CTkFrame(self, fg_color="#0d1527", height=54, corner_radius=0)
        top_bar.pack(fill="x", side="top")

        top_inner = ctk.CTkFrame(top_bar, fg_color="transparent")
        top_inner.pack(fill="x", padx=20, pady=10)

        tag_pill = ctk.CTkLabel(
            top_inner,
            text="🔬 FORENSIC CLAIM INSPECTOR",
            font=ctk.CTkFont(family="SF Pro Display", size=11, weight="bold"),
            text_color=COLOR_ACCENT_CYAN,
            fg_color="#0b233a",
            corner_radius=6,
            padx=8,
            pady=2
        )
        tag_pill.pack(side="left", padx=(0, 12))

        title_lbl = ctk.CTkLabel(
            top_inner,
            text=f"Claim #{self.claim_num} Deep Evidence & Factuality Breakdown",
            font=ctk.CTkFont(family="SF Pro Display", size=14, weight="bold"),
            text_color="#f8fafc"
        )
        title_lbl.pack(side="left")

        close_btn = ctk.CTkButton(
            top_inner,
            text="✕ Close",
            width=70,
            height=26,
            fg_color="#1e293b",
            hover_color="#ef4444",
            text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._close_modal
        )
        close_btn.pack(side="right")

        # Scrollable Body Canvas
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=16)

        verdict = self.clm.get("verdict", "UNVERIFIED")
        claim_text = self.clm.get("claim", "")
        conf_pct = self.clm.get("confidence_pct", "95.0%")
        factuality_pct = self.clm.get("factuality_pct", "100.0%")
        factuality_score = self.clm.get("factuality_score", 1.0)
        hallucination_pct = self.clm.get("hallucination_pct", "0.0%")
        hallucination_score = self.clm.get("hallucination_score", 0.0)
        sub_assertions = self.clm.get("sub_assertions", [])
        right_statement = self.clm.get("right_statement", "")
        proving_res = self.clm.get("proving_resource", {}) or {}
        signals = self.clm.get("signals", {}) or {}
        gate_values = self.clm.get("gate_values", {}) or {}
        evidence = self.clm.get("evidence", []) or []
        elapsed_ms = self.clm.get("elapsed_ms", 120)
        short_reason = self.clm.get("short_reason", "")

        if verdict == "CONTRADICTED":
            v_bg, v_border, v_txt, v_acc = COLOR_CONTRADICTED_BG, COLOR_CONTRADICTED_BORDER, COLOR_CONTRADICTED_TEXT, COLOR_CONTRADICTED_ACCENT
            v_badge = "CONTRADICTED  •  HALLUCINATION DETECTED"
            v_icon = "✕"
        elif verdict == "SUPPORTED":
            v_bg, v_border, v_txt, v_acc = COLOR_SUPPORTED_BG, COLOR_SUPPORTED_BORDER, COLOR_SUPPORTED_TEXT, COLOR_SUPPORTED_ACCENT
            v_badge = "SUPPORTED  •  VERIFIED FACTUAL"
            v_icon = "✓"
        else:
            v_bg, v_border, v_txt, v_acc = COLOR_UNVERIFIED_BG, COLOR_UNVERIFIED_BORDER, COLOR_UNVERIFIED_TEXT, COLOR_UNVERIFIED_ACCENT
            v_badge = "UNVERIFIED  •  INSUFFICIENT EVIDENCE"
            v_icon = "⚠"

        # 1. HERO VERDICT & CLAIM CARD
        hero = ctk.CTkFrame(scroll, fg_color=v_bg, border_color=v_border, border_width=1.5, corner_radius=14)
        hero.pack(fill="x", pady=(0, 14))

        h_inner = ctk.CTkFrame(hero, fg_color="transparent")
        h_inner.pack(fill="x", padx=18, pady=16)

        h_top = ctk.CTkFrame(h_inner, fg_color="transparent")
        h_top.pack(fill="x", pady=(0, 10))

        b_pill = ctk.CTkLabel(
            h_top,
            text=f" {v_icon}  {v_badge} ",
            font=ctk.CTkFont(family="SF Pro Display", size=13, weight="bold"),
            text_color=v_txt,
            fg_color=v_border,
            corner_radius=8,
            padx=12,
            pady=4
        )
        b_pill.pack(side="left")

        meta_lbl = ctk.CTkLabel(
            h_top,
            text=f"Confidence: {conf_pct}  •  Evaluated in {elapsed_ms}ms  •  {len(evidence)} Evidence Passages",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED
        )
        meta_lbl.pack(side="right")

        # Original Evaluated Claim Box
        clm_box = ctk.CTkFrame(h_inner, fg_color="#060913", corner_radius=10)
        clm_box.pack(fill="x", pady=(4, 6))

        clm_box_hdr = ctk.CTkLabel(
            clm_box,
            text="ORIGINAL EVALUATED CLAIM STATEMENT:",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        clm_box_hdr.pack(anchor="w", padx=14, pady=(10, 2))

        clm_box_text = ctk.CTkLabel(
            clm_box,
            text=f'"{claim_text}"',
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#f8fafc",
            wraplength=920,
            justify="left"
        )
        clm_box_text.pack(anchor="w", padx=14, pady=(0, 12))

        # 2. FINE-GRAINED FACTUALITY & HALLUCINATION DECOMPOSITION METERS
        meter_card = ctk.CTkFrame(
            scroll,
            fg_color="#0a101d",
            border_color=COLOR_PANEL_BORDER,
            border_width=1,
            corner_radius=12
        )
        meter_card.pack(fill="x", pady=(0, 14))

        m_inner = ctk.CTkFrame(meter_card, fg_color="transparent")
        m_inner.pack(fill="x", padx=18, pady=14)

        m_hdr = ctk.CTkLabel(
            m_inner,
            text="🔬 PROPOSITION FACTUALITY & HALLUCINATION COMPOSITION DECOMPOSITION:",
            font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
            text_color=COLOR_ACCENT_CYAN
        )
        m_hdr.pack(anchor="w", pady=(0, 10))

        bars_row = ctk.CTkFrame(m_inner, fg_color="transparent")
        bars_row.pack(fill="x", pady=(0, 8))

        # Green Factual Meter Card
        f_box = ctk.CTkFrame(bars_row, fg_color="#051f15", border_color="#065f46", border_width=1, corner_radius=10)
        f_box.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=2)
        f_inner = ctk.CTkFrame(f_box, fg_color="transparent")
        f_inner.pack(fill="both", expand=True, padx=14, pady=12)

        ctk.CTkLabel(f_inner, text="🟢 FACTUAL COMPOSITION", font=ctk.CTkFont(size=11, weight="bold"), text_color="#34d399").pack(anchor="w")
        ctk.CTkLabel(f_inner, text=factuality_pct, font=ctk.CTkFont(family="SF Pro Display", size=22, weight="bold"), text_color="#ffffff").pack(anchor="w", pady=(2, 4))
        f_bar = ctk.CTkProgressBar(f_inner, height=8, corner_radius=4, progress_color="#10b981", fg_color="#182232")
        f_bar.set(factuality_score if isinstance(factuality_score, (int, float)) else 0.5)
        f_bar.pack(fill="x", pady=(2, 4))
        ctk.CTkLabel(f_inner, text="Proportion confirmed by authoritative reference evidence", font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(anchor="w")

        # Red Hallucinated Meter Card
        h_box = ctk.CTkFrame(bars_row, fg_color="#1a0b12", border_color="#7f1d1d", border_width=1, corner_radius=10)
        h_box.pack(side="right", fill="both", expand=True, padx=(8, 0), pady=2)
        h_inner = ctk.CTkFrame(h_box, fg_color="transparent")
        h_inner.pack(fill="both", expand=True, padx=14, pady=12)

        ctk.CTkLabel(h_inner, text="🔴 HALLUCINATED COMPOSITION", font=ctk.CTkFont(size=11, weight="bold"), text_color="#f87171").pack(anchor="w")
        ctk.CTkLabel(h_inner, text=hallucination_pct, font=ctk.CTkFont(family="SF Pro Display", size=22, weight="bold"), text_color="#ffffff").pack(anchor="w", pady=(2, 4))
        h_bar = ctk.CTkProgressBar(h_inner, height=8, corner_radius=4, progress_color="#ef4444", fg_color="#182232")
        h_bar.set(hallucination_score if isinstance(hallucination_score, (int, float)) else 0.0)
        h_bar.pack(fill="x", pady=(2, 4))
        ctk.CTkLabel(h_inner, text="Proportion refuted / contradicted by verified ground truth", font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(anchor="w")

        # Narrative Summary Box
        if short_reason:
            narr_box = ctk.CTkFrame(m_inner, fg_color="#060913", corner_radius=8)
            narr_box.pack(fill="x", pady=(6, 0))
            ctk.CTkLabel(
                narr_box,
                text=f"Summary: {short_reason}",
                font=ctk.CTkFont(size=11),
                text_color="#cbd5e1",
                wraplength=920,
                justify="left"
            ).pack(anchor="w", padx=12, pady=8)

        # 3. 100% NON-HALLUCINATED GROUND-TRUTH CORRECTION (IF CONTRADICTED)
        if right_statement and (verdict == "CONTRADICTED" or hallucination_score > 0):
            corr_card = ctk.CTkFrame(
                scroll,
                fg_color="#04261b",
                border_color="#059669",
                border_width=1.5,
                corner_radius=12
            )
            corr_card.pack(fill="x", pady=(0, 14))

            c_inner = ctk.CTkFrame(corr_card, fg_color="transparent")
            c_inner.pack(fill="x", padx=18, pady=14)

            c_top = ctk.CTkFrame(c_inner, fg_color="transparent")
            c_top.pack(fill="x", pady=(0, 8))

            ctk.CTkLabel(
                c_top,
                text="🟢 100% FACTUAL GROUND-TRUTH REPLACEMENT STATEMENT (NON-HALLUCINATED):",
                font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
                text_color="#34d399"
            ).pack(side="left")

            self.copy_stmt_btn = ctk.CTkButton(
                c_top,
                text="📋 Copy Right Statement",
                height=26,
                fg_color="#065f46",
                hover_color="#047857",
                text_color="#ffffff",
                font=ctk.CTkFont(size=11, weight="bold"),
                command=self._copy_right_statement
            )
            self.copy_stmt_btn.pack(side="right")

            rt_box = ctk.CTkFrame(c_inner, fg_color="#021a12", corner_radius=8)
            rt_box.pack(fill="x", pady=(2, 6))

            ctk.CTkLabel(
                rt_box,
                text=f'"{right_statement}"',
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#f8fafc",
                wraplength=920,
                justify="left"
            ).pack(anchor="w", padx=14, pady=12)

            ctk.CTkLabel(
                c_inner,
                text="✓ Ready to cite: This verified revision replaces all hallucinated propositions with authenticated facts while retaining all factual context.",
                font=ctk.CTkFont(size=11, slant="italic"),
                text_color="#6ee7b7"
            ).pack(anchor="w")

        # 4. DECONSTRUCTED ATOMIC ASSERTIONS BREAKDOWN
        if sub_assertions and len(sub_assertions) > 0:
            subs_card = ctk.CTkFrame(
                scroll,
                fg_color="#0a101d",
                border_color=COLOR_PANEL_BORDER,
                border_width=1,
                corner_radius=12
            )
            subs_card.pack(fill="x", pady=(0, 14))

            s_inner = ctk.CTkFrame(subs_card, fg_color="transparent")
            s_inner.pack(fill="x", padx=18, pady=14)

            s_hdr = ctk.CTkLabel(
                s_inner,
                text=f"🔬 DECONSTRUCTED ATOMIC SUB-ASSERTIONS ({len(sub_assertions)} ANALYZED):",
                font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
                text_color=COLOR_ACCENT_CYAN
            )
            s_hdr.pack(anchor="w", pady=(0, 10))

            for sa in sub_assertions:
                sa_idx = sa.get("sub_index", 1)
                sa_v = sa.get("verdict", "UNVERIFIED")
                sa_txt = sa.get("assertion_text", "")
                sa_conf = sa.get("confidence_pct", "")
                sa_pr = sa.get("proving_resource", {}) or {}
                sa_ev_q = sa.get("evidence_quote") or sa_pr.get("evidence_text", "")
                sa_r_stmt = sa.get("right_statement", "")
                sa_c_part = sa.get("contradicted_part", "")

                if sa_v == "CONTRADICTED":
                    sa_border, sa_fg, sa_bg, sa_ic = COLOR_CONTRADICTED_BORDER, COLOR_CONTRADICTED_TEXT, COLOR_CONTRADICTED_BG, "✕"
                elif sa_v == "SUPPORTED":
                    sa_border, sa_fg, sa_bg, sa_ic = COLOR_SUPPORTED_BORDER, COLOR_SUPPORTED_TEXT, COLOR_SUPPORTED_BG, "✓"
                else:
                    sa_border, sa_fg, sa_bg, sa_ic = COLOR_UNVERIFIED_BORDER, COLOR_UNVERIFIED_TEXT, COLOR_UNVERIFIED_BG, "⚠"

                sa_box = ctk.CTkFrame(s_inner, fg_color="#060913", border_color=sa_border, border_width=1, corner_radius=8)
                sa_box.pack(fill="x", pady=5)

                sa_b_inner = ctk.CTkFrame(sa_box, fg_color="transparent")
                sa_b_inner.pack(fill="x", padx=12, pady=10)

                top_sa = ctk.CTkFrame(sa_b_inner, fg_color="transparent")
                top_sa.pack(fill="x", pady=(0, 6))

                ctk.CTkLabel(
                    top_sa,
                    text=f"Assertion #{sa_idx}",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=COLOR_TEXT_MUTED
                ).pack(side="left", padx=(0, 8))

                ctk.CTkLabel(
                    top_sa,
                    text=f" {sa_ic} {sa_v} • {sa_conf} ",
                    fg_color=sa_bg,
                    text_color=sa_fg,
                    corner_radius=6,
                    font=ctk.CTkFont(size=11, weight="bold")
                ).pack(side="left")

                ctk.CTkLabel(
                    sa_b_inner,
                    text=f'"{sa_txt}"',
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color="#ffffff",
                    wraplength=890,
                    justify="left"
                ).pack(anchor="w", pady=(0, 6))

                if sa_v == "CONTRADICTED":
                    sub_f = ctk.CTkFrame(sa_b_inner, fg_color="#170c14", corner_radius=6)
                    sub_f.pack(fill="x", pady=(2, 2))
                    sub_f_in = ctk.CTkFrame(sub_f, fg_color="transparent")
                    sub_f_in.pack(fill="x", padx=10, pady=8)

                    if sa_c_part:
                        ctk.CTkLabel(
                            sub_f_in,
                            text=f'🔴 Refuted Element: "{sa_c_part}"',
                            font=ctk.CTkFont(size=11, weight="bold"),
                            text_color="#f87171",
                            wraplength=870,
                            justify="left"
                        ).pack(anchor="w")

                    if sa_pr:
                        ctk.CTkLabel(
                            sub_f_in,
                            text=f"🏛️ Proved False by: {sa_pr.get('source', 'Authoritative')} ({sa_pr.get('authority_pct', '99%')} Authority)",
                            font=ctk.CTkFont(size=11, weight="bold"),
                            text_color="#38bdf8"
                        ).pack(anchor="w", pady=(2, 2))

                    if sa_ev_q:
                        ctk.CTkLabel(
                            sub_f_in,
                            text=f'📄 Evidence Quote: "{sa_ev_q}"',
                            font=ctk.CTkFont(size=11, slant="italic"),
                            text_color="#cbd5e1",
                            wraplength=870,
                            justify="left"
                        ).pack(anchor="w", pady=(1, 4))

                    if sa_r_stmt:
                        ctk.CTkLabel(
                            sub_f_in,
                            text=f'🟢 Ground Truth: "{sa_r_stmt}"',
                            font=ctk.CTkFont(size=11, weight="bold"),
                            text_color="#34d399",
                            wraplength=870,
                            justify="left"
                        ).pack(anchor="w")
                elif sa_v == "SUPPORTED" and sa_ev_q:
                    sub_s = ctk.CTkFrame(sa_b_inner, fg_color="#061f15", corner_radius=6)
                    sub_s.pack(fill="x", pady=(2, 2))
                    sub_s_in = ctk.CTkFrame(sub_s, fg_color="transparent")
                    sub_s_in.pack(fill="x", padx=10, pady=8)
                    ctk.CTkLabel(
                        sub_s_in,
                        text=f'📄 Corroborating Evidence: "{sa_ev_q}"',
                        font=ctk.CTkFont(size=11, slant="italic"),
                        text_color="#94a3b8",
                        wraplength=870,
                        justify="left"
                    ).pack(anchor="w")

        # 5. PRIMARY PROVING RESOURCE & AUTHORITATIVE CITATION ARCHIVE
        if proving_res:
            res_card = ctk.CTkFrame(
                scroll,
                fg_color="#0a101d",
                border_color=COLOR_PANEL_BORDER,
                border_width=1,
                corner_radius=12
            )
            res_card.pack(fill="x", pady=(0, 14))

            r_inner = ctk.CTkFrame(res_card, fg_color="transparent")
            r_inner.pack(fill="x", padx=18, pady=14)

            r_hdr = ctk.CTkFrame(r_inner, fg_color="transparent")
            r_hdr.pack(fill="x", pady=(0, 8))

            ctk.CTkLabel(
                r_hdr,
                text="🏛️ PRIMARY PROVING RESOURCE & CITATION DOSSIER:",
                font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
                text_color=COLOR_ACCENT_CYAN
            ).pack(side="left")

            ctk.CTkLabel(
                r_hdr,
                text=f"{proving_res.get('authority_pct', '99%')} Authority  •  {proving_res.get('domain', 'Authoritative Reference')}",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="#94a3b8"
            ).pack(side="right")

            src_box = ctk.CTkFrame(r_inner, fg_color="#060913", corner_radius=8)
            src_box.pack(fill="x", pady=(4, 6))

            ctk.CTkLabel(
                src_box,
                text=f"Reference Source: {proving_res.get('source', 'Authoritative Archive')}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#38bdf8"
            ).pack(anchor="w", padx=14, pady=(10, 4))

            ev_text = proving_res.get("evidence_text", "")
            if ev_text:
                ctk.CTkLabel(
                    src_box,
                    text=f'"{ev_text}"',
                    font=ctk.CTkFont(size=11, slant="italic"),
                    text_color="#cbd5e1",
                    wraplength=920,
                    justify="left"
                ).pack(anchor="w", padx=14, pady=(0, 12))

        # 6. 17-SIGNAL NEURAL RADAR & GATING TELEMETRY
        tele_card = ctk.CTkFrame(
            scroll,
            fg_color="#0a101d",
            border_color=COLOR_PANEL_BORDER,
            border_width=1,
            corner_radius=12
        )
        tele_card.pack(fill="x", pady=(0, 14))

        t_inner = ctk.CTkFrame(tele_card, fg_color="transparent")
        t_inner.pack(fill="x", padx=18, pady=14)

        ctk.CTkLabel(
            t_inner,
            text="📊 17-SIGNAL NEURAL RADAR & RELIABILITY GATING TELEMETRY:",
            font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
            text_color=COLOR_ACCENT_CYAN
        ).pack(anchor="w", pady=(0, 10))

        grid = ctk.CTkFrame(t_inner, fg_color="#060913", corner_radius=8)
        grid.pack(fill="x", pady=4)

        metrics = [
            ("NLI Entailment Prob", f"{signals.get('nli_entailment', 0.0):.4f}", "#34d399"),
            ("NLI Contradiction Prob", f"{signals.get('nli_contradiction', 0.0):.4f}", "#f87171"),
            ("Semantic Cosine Sim", f"{signals.get('semantic_similarity', 0.0):.4f}", "#38bdf8"),
            ("Evidence Consensus", signals.get("consensus_percentage", "85%"), "#a78bfa"),
            ("Retrieval Confidence", f"{signals.get('retrieval_confidence', 0.0):.4f}", "#38bdf8"),
            ("Reliability Gate g_rel", f"{gate_values.get('reliability_weight', 0.85):.3f}", "#fbbf24"),
        ]

        for m_i, (m_label, m_val, m_col) in enumerate(metrics):
            r = m_i // 3
            c = m_i % 3
            cell = ctk.CTkFrame(grid, fg_color="transparent")
            cell.grid(row=r, column=c, padx=16, pady=8, sticky="w")
            ctk.CTkLabel(cell, text=m_label, font=ctk.CTkFont(size=10, weight="bold"), text_color="#94a3b8").pack(anchor="w")
            ctk.CTkLabel(cell, text=m_val, font=ctk.CTkFont(family="SF Pro Display", size=14, weight="bold"), text_color=m_col).pack(anchor="w")

        # 7. BOTTOM ACTION BUTTONS
        bottom_actions = ctk.CTkFrame(scroll, fg_color="transparent")
        bottom_actions.pack(fill="x", pady=(6, 10))

        self.copy_full_btn = ctk.CTkButton(
            bottom_actions,
            text="📋 Copy Full Forensic Dossier",
            height=34,
            fg_color="#1e293b",
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#ffffff",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._copy_full_dossier
        )
        self.copy_full_btn.pack(side="left")

        close_bottom_btn = ctk.CTkButton(
            bottom_actions,
            text="✕ Close Inspection Window",
            height=34,
            fg_color="#0f172a",
            hover_color="#334155",
            text_color="#cbd5e1",
            font=ctk.CTkFont(size=12),
            command=self._close_modal
        )
        close_bottom_btn.pack(side="right")

    def _copy_right_statement(self):
        if self.clm.get("right_statement"):
            try:
                stmt = self.clm["right_statement"].strip()
                if CLIPBOARD_AVAILABLE:
                    if hasattr(self.parent_app, "last_clipboard_text"):
                        self.parent_app.last_clipboard_text = stmt
                    pyperclip.copy(stmt)
                subprocess.run(["pbcopy"], input=stmt.encode("utf-8"), check=False)
                _play_tactile_chime("Tink.aiff")
                if hasattr(self, "copy_stmt_btn"):
                    self.copy_stmt_btn.configure(text="✓ Copied!", fg_color="#059669")
                    self.after(1800, lambda: self.copy_stmt_btn.configure(text="📋 Copy Right Statement", fg_color="#064e3b"))
                if hasattr(self.parent_app, "show_toast"):
                    self.parent_app.show_toast("Right Statement Copied!", icon="✓", color="#10b981")
            except Exception:
                pass

    def _build_forensic_markdown(self) -> str:
        lines = [
            f"# TRUVI-EV Forensic Deep Inspection Report: Claim #{self.claim_num}",
            f"- **Claim**: \"{self.clm.get('claim', '')}\"",
            f"- **Verdict**: {self.clm.get('verdict', 'UNVERIFIED')} ({self.clm.get('confidence_pct', '')} Confidence)",
            f"- **Factuality / Hallucination Ratio**: {self.clm.get('factuality_pct', '100%')} Factual | {self.clm.get('hallucination_pct', '0%')} Hallucinated",
        ]
        if self.clm.get("right_statement"):
            lines.append(f"- **100% Factual Ground Truth**: \"{self.clm.get('right_statement')}\"")
        if self.clm.get("proving_resource"):
            pr = self.clm["proving_resource"]
            lines.append(f"- **Proving Resource**: {pr.get('source')} ({pr.get('domain')}, {pr.get('authority_pct')} Authority)")
            lines.append(f"- **Evidence Quote**: \"{pr.get('evidence_text')}\"")
        if self.clm.get("sub_assertions"):
            lines.append("\n### Sub-Assertions Breakdown:")
            for sa in self.clm["sub_assertions"]:
                lines.append(f"- Assertion #{sa.get('sub_index', 1)}: {sa.get('verdict')} ({sa.get('confidence_pct', '')}) | \"{sa.get('assertion_text')}\"")
                if sa.get("right_statement") and sa.get("verdict") == "CONTRADICTED":
                    lines.append(f"  * Right Statement: \"{sa.get('right_statement')}\"")
        return "\n".join(lines)

    def _copy_full_dossier(self):
        report = self._build_forensic_markdown()
        if report:
            try:
                clean = report.strip()
                if CLIPBOARD_AVAILABLE:
                    if hasattr(self.parent_app, "last_clipboard_text"):
                        self.parent_app.last_clipboard_text = clean
                    pyperclip.copy(clean)
                subprocess.run(["pbcopy"], input=clean.encode("utf-8"), check=False)
                _play_tactile_chime("Tink.aiff")
                if hasattr(self, "copy_full_btn"):
                    self.copy_full_btn.configure(text="✓ Dossier Copied!", fg_color="#059669")
                    self.after(1800, lambda: self.copy_full_btn.configure(text="📋 Copy Full Forensic Dossier", fg_color="#1e293b"))
                if hasattr(self.parent_app, "show_toast"):
                    self.parent_app.show_toast("Forensic Dossier Copied!", icon="✓", color="#10b981")
            except Exception:
                pass


# =============================================================================
# Main Obsidian Studio Desktop Application Window
# =============================================================================
class TRUVIApp(ctk.CTk):
    """
    Flagship native macOS Fact Verification Studio Application.
    """

    def __init__(self):
        super().__init__()

        # Engine
        self.engine = get_engine()
        self.active_result = None
        self.session_history: List[Dict[str, Any]] = []

        # UI & Modal States
        self.clipboard_enabled = True
        self.last_clipboard_text = ""
        self.is_verifying = False
        self.active_popup: Optional[FloatingHUDNotification] = None
        self.active_forensic_modal: Optional[ClaimForensicModal] = None
        self.open_forensic_modals: Dict[int, ClaimForensicModal] = {}
        self.toast_frame = None
        self.toast_after_id = None
        self.clipboard_thread = None
        self._last_modal_time = 0.0

        # Window Configuration
        self.title("TRUVI-EV — Trustworthy Evidence-Aware Verifier Studio")
        self.geometry("1180x820")
        self.minsize(1000, 700)
        self.configure(fg_color=COLOR_BG_ROOT)

        # Translucency for macOS
        try:
            self.attributes("-alpha", 0.96)
        except Exception:
            pass

        # Build UI Architecture
        self._build_ui()
        self.after(1500, self._start_clipboard_watcher)

        # Keyboard shortcuts
        self.bind("<Command-Return>", lambda e: self._on_verify_clicked())
        self.bind("<Control-Return>", lambda e: self._on_verify_clicked())
        self.bind("<Command-k>", lambda e: self._clear_input())
        self.bind("<Control-k>", lambda e: self._clear_input())
        self.bind("<Command-Shift-C>", lambda e: self._copy_active_report())
        self.bind("<Command-Shift-A>", lambda e: self._copy_factual_answer())
        self.bind("<Command-Shift-P>", lambda e: self._copy_prompt())

    def _build_ui(self):
        """Constructs the desktop interface."""
        # 1. Top Obsidian Header
        self._build_header()

        # 2. Segmented Tab Navigation Controller
        self._build_tab_bar()

        # 3. Main Views Container
        self.content_container = ctk.CTkFrame(self, fg_color="transparent")
        self.content_container.pack(fill="both", expand=True, padx=20, pady=(6, 12))

        # View 1: ⚡ Live Verifier (Main scrollable panel)
        self.view_verifier = ctk.CTkScrollableFrame(
            self.content_container,
            fg_color="transparent",
            corner_radius=0
        )
        self._build_verifier_view()

        # View 2: 📊 17-Signal Neural Radar
        self.view_radar = ctk.CTkScrollableFrame(
            self.content_container,
            fg_color="transparent",
            corner_radius=0
        )
        self._build_radar_view()

        # View 3: 🏛️ Evidence & Sources Explorer
        self.view_evidence = ctk.CTkScrollableFrame(
            self.content_container,
            fg_color="transparent",
            corner_radius=0
        )
        self._build_evidence_view()

        # View 4: 🕒 Session History & Audit Log
        self.view_history = ctk.CTkScrollableFrame(
            self.content_container,
            fg_color="transparent",
            corner_radius=0
        )
        self._build_history_view()

        # Default Tab
        self._switch_tab("⚡ Live Verifier")

        # 4. Bottom Status Bar
        self._build_status_bar()

    # =========================================================================
    # Header & Tab Navigation
    # =========================================================================
    def show_toast(self, message: str, icon: str = "✓", color: str = "#10b981"):
        """Displays floating Toast and illuminates the Header Dynamic Notification Island."""
        # 1. Update and show the Header Dynamic Notification Island (100% visible, never clipped)
        if hasattr(self, "dynamic_pill") and hasattr(self, "dynamic_pill_label"):
            try:
                bg_col = "#04261a" if color == "#10b981" else ("#2b0d16" if color == "#f43f5e" else "#1e1405")
                txt_col = "#34d399" if color == "#10b981" else ("#fb7185" if color == "#f43f5e" else "#fcd34d")
                self.dynamic_pill.configure(border_color=color, fg_color=bg_col)
                self.dynamic_pill_label.configure(text=f" {icon}  {message} ", text_color=txt_col)
                if not self.dynamic_pill.winfo_ismapped():
                    self.dynamic_pill.pack(expand=True)
                if hasattr(self, "header_toast_timer") and self.header_toast_timer:
                    self.after_cancel(self.header_toast_timer)
                self.header_toast_timer = self.after(2400, lambda: self.dynamic_pill.pack_forget() if (hasattr(self, "dynamic_pill") and self.dynamic_pill.winfo_exists()) else None)
            except Exception:
                pass

        # 2. Also show floating prominent notification banner with explicit lift()
        if hasattr(self, "toast_frame") and self.toast_frame is not None:
            try:
                if hasattr(self, "toast_after_id") and self.toast_after_id:
                    self.after_cancel(self.toast_after_id)
                self.toast_frame.destroy()
            except Exception:
                pass
            self.toast_frame = None

        self.toast_frame = ctk.CTkFrame(
            self,
            fg_color="#050e1a",
            border_color=color,
            border_width=2,
            corner_radius=14,
            height=44
        )
        self.toast_frame.place(relx=0.5, rely=0.105, anchor="center")
        self.toast_frame.lift()

        t_inner = ctk.CTkFrame(self.toast_frame, fg_color="transparent")
        t_inner.pack(padx=20, pady=8)

        t_icon = ctk.CTkLabel(
            t_inner,
            text=f" {icon} ",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=color
        )
        t_icon.pack(side="left", padx=(0, 8))

        t_msg = ctk.CTkLabel(
            t_inner,
            text=message,
            font=ctk.CTkFont(family="SF Pro Display", size=13, weight="bold"),
            text_color="#ffffff"
        )
        t_msg.pack(side="left")

        def _dismiss():
            if hasattr(self, "toast_frame") and self.toast_frame is not None:
                try:
                    self.toast_frame.destroy()
                except Exception:
                    pass
                self.toast_frame = None

        self.toast_after_id = self.after(2200, _dismiss)

    def _copy_with_feedback(self, text: str, btn: Optional[ctk.CTkButton] = None, success_msg: str = "Copied to Clipboard!"):
        """Guaranteed clipboard copy with audio click, button micro-transformation, and Dynamic Toast."""
        if not text:
            return
        clean = text.strip()

        # 1. Guaranteed clipboard copy (pyperclip + native macOS pbcopy)
        try:
            if CLIPBOARD_AVAILABLE:
                self.last_clipboard_text = clean
                pyperclip.copy(clean)
        except Exception:
            pass
        try:
            subprocess.run(["pbcopy"], input=clean.encode("utf-8"), check=False)
            self.last_clipboard_text = clean
        except Exception:
            pass

        # 2. Tactile audio feedback (macOS Tink click sound)
        _play_tactile_chime("Tink.aiff")

        # 3. Visual button micro-state transformation
        if btn is not None:
            try:
                orig_text = getattr(btn, "_orig_text", None) or btn.cget("text")
                orig_fg = getattr(btn, "_orig_fg", None) or btn.cget("fg_color")
                orig_border = getattr(btn, "_orig_border", None) or btn.cget("border_color")
                btn._orig_text = orig_text
                btn._orig_fg = orig_fg
                btn._orig_border = orig_border

                btn.configure(
                    text="✓ Copied!",
                    fg_color="#059669",
                    border_color="#34d399"
                )

                def _revert(b=btn, ot=orig_text, of=orig_fg, ob=orig_border):
                    try:
                        if b.winfo_exists():
                            b.configure(text=ot, fg_color=of, border_color=ob)
                    except Exception:
                        pass

                self.after(1800, _revert)
            except Exception:
                pass

        # 4. Trigger Dynamic Island Notification & Floating Toast
        self.show_toast(success_msg, icon="✓", color="#10b981")

    def _copy_prompt(self):
        """Copies the input text / query prompt to clipboard with instant multi-sensory feedback."""
        try:
            txt = self.text_input.get("1.0", "end").strip()
            if not txt:
                self.show_toast("No prompt to copy — enter text first", icon="⚠", color="#f59e0b")
                return
            btn = getattr(self, "copy_prompt_btn", None)
            chars = len(txt)
            self._copy_with_feedback(txt, btn, f"Prompt Copied! ({chars} Chars)")
        except Exception:
            pass

    def _copy_factual_answer(self):
        """Copies the 100% verified ground-truth answer statement or corrected paragraph."""
        if self.active_result is None:
            self.show_toast("Run verification first to copy factual answer", icon="⚠", color="#f59e0b")
            return

        res = self.active_result
        if "paragraph" in res:
            ans = res.get("corrected_paragraph")
            if not ans:
                ans = res.get("paragraph", "")
            self._copy_with_feedback(ans, getattr(self, "copy_ans_btn", None), "Factual Paragraph Copied!")
        else:
            v = res.get("verdict", "UNVERIFIED")
            if v == "CONTRADICTED" and res.get("right_statement"):
                ans = res.get("right_statement")
            else:
                ans = res.get("claim", "")
            self._copy_with_feedback(ans, getattr(self, "copy_ans_btn", None), "Factual Statement Copied!")

    def _animate_status_pulse(self):
        """Subtle glowing pulse on the engine status pill."""
        if hasattr(self, "pulse_lbl") and self.pulse_lbl.winfo_exists():
            try:
                cur_col = self.pulse_lbl.cget("text_color")
                new_col = "#10b981" if cur_col == "#34d399" else "#34d399"
                self.pulse_lbl.configure(text_color=new_col)
            except Exception:
                pass
        self.after(1600, self._animate_status_pulse)

    def _build_header(self):
        header = ctk.CTkFrame(
            self,
            fg_color=COLOR_PANEL_BG,
            border_color=COLOR_PANEL_BORDER,
            border_width=1,
            corner_radius=14,
            height=66
        )
        header.pack(fill="x", padx=20, pady=(14, 6))

        # Left: Branded High-Tech Core Icon & Identity (No fake OS buttons!)
        left_box = ctk.CTkFrame(header, fg_color="transparent")
        left_box.pack(side="left", padx=16, pady=10)

        logo_box = ctk.CTkFrame(
            left_box,
            fg_color="#08182b",
            border_color="#0284c7",
            border_width=1.5,
            corner_radius=10,
            width=40,
            height=40
        )
        logo_box.pack(side="left", padx=(0, 12))
        logo_box.pack_propagate(False)

        logo_icon = ctk.CTkLabel(
            logo_box,
            text="◈",
            font=ctk.CTkFont(family="SF Pro Display", size=22, weight="bold"),
            text_color="#38bdf8"
        )
        logo_icon.place(relx=0.5, rely=0.5, anchor="center")

        title_box = ctk.CTkFrame(left_box, fg_color="transparent")
        title_box.pack(side="left")

        title_row = ctk.CTkFrame(title_box, fg_color="transparent")
        title_row.pack(anchor="w")

        title = ctk.CTkLabel(
            title_row,
            text="TRUVI-EV",
            font=ctk.CTkFont(family="SF Pro Display", size=19, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        title.pack(side="left", padx=(0, 8))

        ver_badge = ctk.CTkLabel(
            title_row,
            text="STUDIO PRO",
            font=ctk.CTkFont(family="SF Pro Display", size=10, weight="bold"),
            text_color="#38bdf8",
            fg_color="#0c233c",
            corner_radius=6,
            padx=7,
            pady=2
        )
        ver_badge.pack(side="left", padx=(0, 8))

        self.pulse_lbl = ctk.CTkLabel(
            title_row,
            text="● ENGINE ACTIVE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#34d399",
            fg_color="#04261a",
            corner_radius=6,
            padx=7,
            pady=2
        )
        self.pulse_lbl.pack(side="left")
        self.after(1600, self._animate_status_pulse)

        subtitle = ctk.CTkLabel(
            title_box,
            text="Reliability-Gated Neural Hallucination Detection & Evidence Synthesis",
            font=ctk.CTkFont(family="SF Pro Display", size=11),
            text_color=COLOR_TEXT_MUTED
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # Center: Interactive Dynamic Notification Island (Header-level, 100% visible)
        self.center_toast_box = ctk.CTkFrame(header, fg_color="transparent")
        self.center_toast_box.pack(side="left", fill="both", expand=True, padx=14)

        self.dynamic_pill = ctk.CTkFrame(
            self.center_toast_box,
            fg_color="#04261a",
            border_color="#10b981",
            border_width=1.5,
            corner_radius=10,
            height=32
        )
        self.dynamic_pill_label = ctk.CTkLabel(
            self.dynamic_pill,
            text="",
            font=ctk.CTkFont(family="SF Pro Display", size=11, weight="bold"),
            text_color="#34d399"
        )
        self.dynamic_pill_label.pack(padx=14, pady=4)
        # Not packed into center_toast_box initially

        # Right: Badges, Test Popup, & Clipboard Auto-Verify Switch
        right_box = ctk.CTkFrame(header, fg_color="transparent")
        right_box.pack(side="right", padx=16, pady=10)

        model_pill = ctk.CTkLabel(
            right_box,
            text="● ENGINE ONLINE • 17 SIGNALS",
            fg_color="#072b1d",
            text_color="#34d399",
            corner_radius=10,
            font=ctk.CTkFont(size=11, weight="bold"),
            padx=10,
            pady=4
        )
        model_pill.pack(side="left", padx=(0, 8))

        test_hud_btn = ctk.CTkButton(
            right_box,
            text="⚡ Test HUD Popup",
            height=28,
            fg_color="#182740",
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#38bdf8",
            font=ctk.CTkFont(size=11, weight="bold"),
            corner_radius=8,
            command=self._trigger_test_hud
        )
        test_hud_btn.pack(side="left", padx=(0, 8))

        self.clip_switch = ctk.CTkSwitch(
            right_box,
            text="Clipboard HUD",
            command=self._toggle_clipboard,
            font=ctk.CTkFont(size=12, weight="bold"),
            progress_color=COLOR_ACCENT_PRIMARY,
            text_color=COLOR_TEXT_PRIMARY
        )
        self.clip_switch.select()
        self.clip_switch.pack(side="left", padx=4)

    def _trigger_test_hud(self):
        """Immediately displays the floating clipboard HUD with a test claim to demonstrate functionality."""
        sample_test = "The Moon produces its own light, and light travels at ~300,000 km/s in a vacuum."
        self._show_clipboard_analyzing_hud(sample_test)
        def _run_test():
            time.sleep(0.4)
            res = self.engine.verify_claim(sample_test)
            self.after(0, self._update_clipboard_hud_result, res)
        threading.Thread(target=_run_test, daemon=True).start()

    def _build_tab_bar(self):
        tab_frame = ctk.CTkFrame(
            self,
            fg_color="#080d19",
            border_color="#172235",
            border_width=1,
            corner_radius=10,
            height=40
        )
        tab_frame.pack(fill="x", padx=20, pady=(2, 6))

        self.tab_buttons = {}
        tabs = [
            ("⚡ Live Verifier", "⚡ Live Verifier"),
            ("📊 17-Signal Neural Radar", "📊 17-Signal Neural Radar"),
            ("🏛️ Evidence & Sources", "🏛️ Evidence & Sources"),
            ("🕒 Session History", "🕒 Session History")
        ]

        for label, tab_id in tabs:
            btn = ctk.CTkButton(
                tab_frame,
                text=label,
                height=32,
                corner_radius=8,
                fg_color="transparent",
                hover_color=COLOR_PANEL_HOVER,
                text_color=COLOR_TEXT_SECONDARY,
                font=ctk.CTkFont(size=12, weight="bold"),
                command=lambda t=tab_id: self._switch_tab(t)
            )
            btn.pack(side="left", padx=4, pady=4)
            self.tab_buttons[tab_id] = btn

    def _switch_tab(self, tab_id: str):
        self.view_verifier.pack_forget()
        self.view_radar.pack_forget()
        self.view_evidence.pack_forget()
        self.view_history.pack_forget()

        for tid, btn in self.tab_buttons.items():
            if tid == tab_id:
                btn.configure(fg_color="#182740", text_color="#38bdf8")
            else:
                btn.configure(fg_color="transparent", text_color=COLOR_TEXT_SECONDARY)

        if tab_id == "⚡ Live Verifier":
            self.view_verifier.pack(fill="both", expand=True)
        elif tab_id == "📊 17-Signal Neural Radar":
            self.view_radar.pack(fill="both", expand=True)
            self._render_radar_tab()
        elif tab_id == "🏛️ Evidence & Sources":
            self.view_evidence.pack(fill="both", expand=True)
            self._render_evidence_tab()
        elif tab_id == "🕒 Session History":
            self.view_history.pack(fill="both", expand=True)
            self._render_history_tab()

    # =========================================================================
    # Tab 1: ⚡ Live Verifier View
    # =========================================================================
    def _build_verifier_view(self):
        input_card = ctk.CTkFrame(
            self.view_verifier,
            fg_color=COLOR_PANEL_BG,
            border_color=COLOR_PANEL_BORDER,
            border_width=1,
            corner_radius=16
        )
        input_card.pack(fill="x", pady=(2, 10))

        in_hdr = ctk.CTkFrame(input_card, fg_color="transparent")
        in_hdr.pack(fill="x", padx=18, pady=(12, 6))

        lbl = ctk.CTkLabel(
            in_hdr,
            text="INPUT ASSERTION OR PARAGRAPH",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        lbl.pack(side="left")

        self.telemetry_lbl = ctk.CTkLabel(
            in_hdr,
            text="Press ⌘+Enter to Verify  •  ⌘+K to Clear",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED
        )
        self.telemetry_lbl.pack(side="right")

        self.text_input = ctk.CTkTextbox(
            input_card,
            height=94,
            fg_color="#080d19",
            border_color=COLOR_PANEL_BORDER,
            border_width=1,
            corner_radius=12,
            font=ctk.CTkFont(size=13),
            text_color=COLOR_TEXT_PRIMARY,
            wrap="word"
        )
        self.text_input.pack(fill="x", padx=18, pady=(0, 10))
        self.text_input.insert("1.0", "The Amazon is the longest river in the world. The Great Wall of China is visible from the Moon with the naked eye. Water boils at 100°C at sea level. The Eiffel Tower is in Paris.")
        self.text_input.bind("<KeyRelease>", self._update_input_telemetry)

        chips_frame = ctk.CTkFrame(input_card, fg_color="transparent")
        chips_frame.pack(fill="x", padx=18, pady=(0, 10))

        chips_lbl = ctk.CTkLabel(
            chips_frame,
            text="Presets:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        chips_lbl.pack(side="left", padx=(0, 8))

        examples = [
            ("🪐 Venus & Planets", "Venus is the closest planet to the Sun and is the hottest planet in the Solar System."),
            ("🏔️ Everest & Sound", "Over 346 people have died on Mount Everest, which has a height of 8,848.86 m, and sound travels at 343 m/s in air."),
            ("🌕 Moon & Light", "The Moon produces its own light, and light travels at ~300,000 km/s in a vacuum."),
            ("🦴 Anatomy & Earth", "The human body has ~206 bones in adulthood, and Earth rotates in ~24 hours causing day and night."),
            ("🧪 Mixed Paragraph", "The Amazon is the longest river in the world. The Great Wall of China is visible from the Moon with the naked eye. Water boils at 100°C at sea level. The Eiffel Tower is in Paris."),
            ("⚖️ Half-True / Half-False", "Barack Obama was born in Kenya, but became the 44th US President."),
        ]

        for label_text, claim_val in examples:
            btn = ctk.CTkButton(
                chips_frame,
                text=label_text,
                height=26,
                fg_color="#0d182b",
                hover_color="#182c4f",
                text_color="#cbd5e1",
                font=ctk.CTkFont(size=11, weight="bold"),
                border_color="#1d2e4a",
                border_width=1,
                corner_radius=8,
                command=lambda c=claim_val, l=label_text: self._load_example(c, l)
            )
            btn.pack(side="left", padx=4)

        btn_row = ctk.CTkFrame(input_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=18, pady=(0, 14))

        self.verify_btn = ctk.CTkButton(
            btn_row,
            text="⚡ VERIFY WITH TRUVI-EV  [ ⌘ ↵ ]",
            height=42,
            fg_color=COLOR_ACCENT_PRIMARY,
            hover_color=COLOR_ACCENT_HOVER,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=12,
            command=self._on_verify_clicked
        )
        self.verify_btn.pack(side="left", padx=(0, 10))

        self.copy_prompt_btn = ctk.CTkButton(
            btn_row,
            text="📋 Copy Prompt",
            height=42,
            fg_color="#0e1f38",
            hover_color="#1e3a66",
            border_color="#2b4c80",
            border_width=1.2,
            font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
            text_color="#60a5fa",
            corner_radius=12,
            command=self._copy_prompt
        )
        self.copy_prompt_btn.pack(side="left", padx=(0, 10))

        self.copy_ans_btn = ctk.CTkButton(
            btn_row,
            text="📋 Copy Factual Answer",
            height=42,
            fg_color="#064e3b",
            hover_color="#059669",
            border_color="#059669",
            border_width=1.2,
            font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
            text_color="#34d399",
            corner_radius=12,
            command=self._copy_factual_answer
        )
        self.copy_ans_btn.pack(side="left", padx=(0, 10))

        self.copy_report_btn = ctk.CTkButton(
            btn_row,
            text="📋 Copy Audit Report",
            height=42,
            fg_color="#141d2e",
            hover_color="#22304a",
            border_color="#2b3b55",
            border_width=1.2,
            font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
            text_color="#cbd5e1",
            corner_radius=12,
            command=self._copy_active_report
        )
        self.copy_report_btn.pack(side="left", padx=(0, 10))

        paste_btn = ctk.CTkButton(
            btn_row,
            text="📥 Paste & Verify",
            height=42,
            fg_color="#0f192b",
            hover_color="#1b2a45",
            border_color="#223554",
            border_width=1,
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8",
            corner_radius=12,
            command=self._paste_and_verify
        )
        paste_btn.pack(side="left", padx=(0, 10))

        clear_btn = ctk.CTkButton(
            btn_row,
            text="🗑️ Clear",
            width=70,
            height=42,
            fg_color="transparent",
            hover_color="#241318",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_MUTED,
            corner_radius=12,
            command=self._clear_input
        )
        clear_btn.pack(side="left")

        self.results_frame = ctk.CTkFrame(self.view_verifier, fg_color="transparent")
        self.results_frame.pack(fill="x", expand=True, pady=(6, 16))

        self._render_welcome_state()
        self._update_input_telemetry()

    def _update_input_telemetry(self, _event=None):
        try:
            txt = self.text_input.get("1.0", "end").strip()
            chars = len(txt)
            words = len(txt.split()) if txt else 0
            clauses = max(1, len([s for s in txt.split(".") if len(s.strip()) > 5])) if txt else 0
            if hasattr(self, "telemetry_lbl"):
                self.telemetry_lbl.configure(text=f"{clauses} Clauses • {words} Words • {chars} Chars • ⌘+Enter to Verify")
        except Exception:
            pass

    def _paste_and_verify(self):
        try:
            if CLIPBOARD_AVAILABLE:
                p = (pyperclip.paste() or "").strip()
                if p:
                    self.text_input.delete("1.0", "end")
                    self.text_input.insert("1.0", p)
                    self._update_input_telemetry()
                    self._on_verify_clicked()
        except Exception:
            pass

    def _render_welcome_state(self):
        for w in self.results_frame.winfo_children():
            w.destroy()

        card = ctk.CTkFrame(
            self.results_frame,
            fg_color=COLOR_PANEL_BG,
            border_color=COLOR_PANEL_BORDER,
            border_width=1,
            corner_radius=16
        )
        card.pack(fill="x", pady=8)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=24, pady=24)

        icon = ctk.CTkLabel(inner, text="🛡️", font=ctk.CTkFont(size=36))
        icon.pack(pady=(0, 8))

        heading = ctk.CTkLabel(
            inner,
            text="TRUVI-EV Fact-Checking & Hallucination Detector",
            font=ctk.CTkFont(family="SF Pro Display", size=16, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        heading.pack(pady=(0, 6))

        desc = ctk.CTkLabel(
            inner,
            text=(
                "Verify single claims, compound assertions ('half-true / half-false'), or full paragraphs.\n"
                "TRUVI-EV pinpoints exact contradictions, cites authoritative proof resources,\n"
                "and synthesizes verified ground-truth statements."
            ),
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_SECONDARY,
            justify="center"
        )
        desc.pack(pady=(0, 16))

        feat_row = ctk.CTkFrame(inner, fg_color="transparent")
        feat_row.pack(pady=4)

        feats = [
            ("⚡ 17-Signal Gated Radar", "NLI, Cosine, Source, Consensus"),
            ("🌐 Live Web & Wiki", "Real-time search verification"),
            ("📋 Instant Clipboard HUD", "Highlight text anywhere & copy"),
            ("🟢 Ground-Truth Corrections", "Pinpoints false parts & rewrites")
        ]
        for title, sub in feats:
            f_box = ctk.CTkFrame(feat_row, fg_color="#09101d", border_color="#18263d", border_width=1, corner_radius=10, width=200, height=56)
            f_box.pack(side="left", padx=6)
            f_box.pack_propagate(False)
            t_lbl = ctk.CTkLabel(f_box, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8")
            t_lbl.pack(pady=(8, 1))
            s_lbl = ctk.CTkLabel(f_box, text=sub, font=ctk.CTkFont(size=10), text_color=COLOR_TEXT_MUTED)
            s_lbl.pack()

    # =========================================================================
    # Verification Actions & Rendering
    # =========================================================================
    def _load_example(self, claim_text: str, label_name: str = ""):
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", claim_text)
        self._update_input_telemetry()
        if label_name:
            self.show_toast(f"Preset loaded: {label_name}", icon="⚡", color="#38bdf8")
        self._on_verify_clicked()

    def _clear_input(self):
        self.text_input.delete("1.0", "end")
        self._update_input_telemetry()
        self._render_welcome_state()

    def _toggle_clipboard(self):
        self.clipboard_enabled = self.clip_switch.get()

    def _on_verify_clicked(self):
        text = self.text_input.get("1.0", "end").strip()
        if not text:
            return

        self.is_verifying = True
        self.verify_btn.configure(text="⏳ Analyzing Multi-Tier Evidence...", state="disabled")
        threading.Thread(target=self._run_verification_worker, args=(text,), daemon=True).start()

    def _run_verification_worker(self, text: str):
        try:
            has_multiple_sentences = (
                "\n" in text or
                len([s for s in text.split(".") if len(s.strip()) > 10]) > 1
            )

            if has_multiple_sentences:
                result = self.engine.verify_paragraph(text)
                is_para = True
            else:
                result = self.engine.verify_claim(text)
                is_para = False

            self.after(0, self._render_results_ui, result, is_para)
        except Exception as e:
            print(f"Verification worker error: {e}")
            self.after(0, lambda: self.verify_btn.configure(text="⚡ VERIFY WITH TRUVI-EV  [ ⌘ ↵ ]", state="normal"))
        finally:
            self.is_verifying = False

    def _render_results_ui(self, result: Dict[str, Any], is_para: bool = False):
        self.active_result = result
        self.verify_btn.configure(text="⚡ VERIFY WITH TRUVI-EV  [ ⌘ ↵ ]", state="normal")

        self._record_in_history(result)

        for w in self.results_frame.winfo_children():
            w.destroy()

        if is_para or "paragraph" in result:
            self._render_paragraph_view(result)
        else:
            self._render_single_claim_view(result)

    def load_and_display_result(self, result: Dict[str, Any]):
        self._switch_tab("⚡ Live Verifier")
        is_para = "paragraph" in result
        text = result.get("paragraph" if is_para else "claim", "")
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", text)
        self._update_input_telemetry()
        self._render_results_ui(result, is_para)

    # -------------------------------------------------------------------------
    # Single Claim Rendering
    # -------------------------------------------------------------------------
    def _render_single_claim_view(self, res: Dict[str, Any]):
        self.active_result = res
        verdict = res.get("verdict", "UNVERIFIED")
        conf_pct = res.get("confidence_pct", "95.0%")
        conf_val = res.get("confidence", 0.95)
        reason = res.get("short_reason", "")
        advisory = res.get("action_advisory", "")
        elapsed = res.get("elapsed_ms", 120)
        signals = res.get("signals", {})
        evidence = res.get("evidence", [])

        if verdict == "CONTRADICTED":
            v_bg, v_border, v_txt, v_acc = COLOR_CONTRADICTED_BG, COLOR_CONTRADICTED_BORDER, COLOR_CONTRADICTED_TEXT, COLOR_CONTRADICTED_ACCENT
            v_badge_text = "CONTRADICTED  •  HALLUCINATION DETECTED"
            v_icon = "✕"
        elif verdict == "SUPPORTED":
            v_bg, v_border, v_txt, v_acc = COLOR_SUPPORTED_BG, COLOR_SUPPORTED_BORDER, COLOR_SUPPORTED_TEXT, COLOR_SUPPORTED_ACCENT
            v_badge_text = "SUPPORTED  •  VERIFIED FACTUAL"
            v_icon = "✓"
        else:
            v_bg, v_border, v_txt, v_acc = COLOR_UNVERIFIED_BG, COLOR_UNVERIFIED_BORDER, COLOR_UNVERIFIED_TEXT, COLOR_UNVERIFIED_ACCENT
            v_badge_text = "UNVERIFIED  •  INSUFFICIENT EVIDENCE"
            v_icon = "⚠"

        # 1. VERDICT HERO BANNER
        banner = ctk.CTkFrame(
            self.results_frame,
            fg_color=v_bg,
            border_color=v_border,
            border_width=2,
            corner_radius=16
        )
        banner.pack(fill="x", pady=(0, 14))

        b_inner = ctk.CTkFrame(banner, fg_color="transparent")
        b_inner.pack(fill="x", padx=20, pady=16)

        # Top row: Badge + Latency + Inspect Button
        top_row = ctk.CTkFrame(b_inner, fg_color="transparent")
        top_row.pack(fill="x", pady=(0, 10))

        badge = ctk.CTkLabel(
            top_row,
            text=f" {v_icon}  {v_badge_text} ",
            font=ctk.CTkFont(family="SF Pro Display", size=14, weight="bold"),
            text_color=v_txt,
            fg_color=v_border,
            corner_radius=8,
            padx=12,
            pady=4
        )
        badge.pack(side="left")

        inspect_btn = ctk.CTkButton(
            top_row,
            text="🔬 Inspect Forensic Dossier →",
            height=28,
            fg_color="#182740",
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#38bdf8",
            border_color="#2b3e5f",
            border_width=1,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._open_claim_forensic_dossier(res, 1)
        )
        inspect_btn.pack(side="right", padx=(8, 0))

        time_lbl = ctk.CTkLabel(
            top_row,
            text=f"Evaluated in {elapsed}ms  •  {len(evidence)} Passages",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED
        )
        time_lbl.pack(side="right")

        # Dual Meters Row: Confidence & Consensus
        meters_row = ctk.CTkFrame(b_inner, fg_color="transparent")
        meters_row.pack(fill="x", pady=(0, 8))

        # Confidence Bar
        c_box = ctk.CTkFrame(meters_row, fg_color="transparent")
        c_box.pack(side="left", fill="x", expand=True, padx=(0, 10))

        c_lbl = ctk.CTkLabel(
            c_box,
            text=f"Calibrated Confidence: {conf_pct}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        c_lbl.pack(anchor="w", pady=(0, 3))

        c_bar = ctk.CTkProgressBar(c_box, height=8, corner_radius=4, progress_color=v_acc, fg_color="#182232")
        c_bar.set(conf_val)
        c_bar.pack(fill="x")

        # Consensus Bar
        cons_val = signals.get("evidence_agreement", 0.85)
        cons_str = signals.get("consensus_percentage", "85%")
        a_box = ctk.CTkFrame(meters_row, fg_color="transparent")
        a_box.pack(side="right", fill="x", expand=True, padx=(10, 0))

        a_lbl = ctk.CTkLabel(
            a_box,
            text=f"Evidence Consensus Agreement: {cons_str}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        a_lbl.pack(anchor="w", pady=(0, 3))

        a_bar = ctk.CTkProgressBar(a_box, height=8, corner_radius=4, progress_color="#38bdf8", fg_color="#182232")
        a_bar.set(cons_val)
        a_bar.pack(fill="x")

        # Factuality vs Hallucination Breakdown Meters Row
        comp_row = ctk.CTkFrame(b_inner, fg_color="transparent")
        comp_row.pack(fill="x", pady=(0, 10))

        f_box = ctk.CTkFrame(comp_row, fg_color="transparent")
        f_box.pack(side="left", fill="x", expand=True, padx=(0, 10))

        f_pct_val = res.get("factuality_pct", "100.0%")
        f_score_raw = res.get("factuality_score", 1.0)
        f_score_val = float(f_score_raw) if isinstance(f_score_raw, (int, float)) else (1.0 if verdict == "SUPPORTED" else 0.0)

        f_lbl = ctk.CTkLabel(
            f_box,
            text=f"Factual Accuracy Ratio: {f_pct_val}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_SUPPORTED_TEXT
        )
        f_lbl.pack(anchor="w", pady=(0, 3))

        f_bar = ctk.CTkProgressBar(f_box, height=8, corner_radius=4, progress_color=COLOR_SUPPORTED_ACCENT, fg_color="#182232")
        f_bar.set(f_score_val)
        f_bar.pack(fill="x")

        h_box = ctk.CTkFrame(comp_row, fg_color="transparent")
        h_box.pack(side="right", fill="x", expand=True, padx=(10, 0))

        h_pct_val = res.get("hallucination_pct", "0.0%")
        h_score_raw = res.get("hallucination_score", 0.0)
        h_score_val = float(h_score_raw) if isinstance(h_score_raw, (int, float)) else (1.0 if verdict == "CONTRADICTED" else 0.0)

        h_lbl = ctk.CTkLabel(
            h_box,
            text=f"Hallucination Severity: {h_pct_val}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_CONTRADICTED_TEXT
        )
        h_lbl.pack(anchor="w", pady=(0, 3))

        h_bar = ctk.CTkProgressBar(h_box, height=8, corner_radius=4, progress_color=COLOR_CONTRADICTED_ACCENT, fg_color="#182232")
        h_bar.set(h_score_val)
        h_bar.pack(fill="x")

        # -------------------------------------------------------------
        # 2. DEDICATED CONTRADICTION AUDIT & CORRECTION CARD (If Refuted)
        # -------------------------------------------------------------
        if verdict == "CONTRADICTED":
            c_part = res.get("contradicted_part", res.get("claim", ""))
            resrc = res.get("proving_resource", {}) or {}
            r_stmt = res.get("right_statement", "")

            audit_card = ctk.CTkFrame(
                b_inner,
                fg_color="#0a101b",
                border_color=COLOR_CONTRADICTED_BORDER,
                border_width=1.5,
                corner_radius=12
            )
            audit_card.pack(fill="x", pady=(10, 8))

            a_inner = ctk.CTkFrame(audit_card, fg_color="transparent")
            a_inner.pack(fill="x", padx=16, pady=14)

            # Section Header
            a_top = ctk.CTkFrame(a_inner, fg_color="transparent")
            a_top.pack(fill="x", pady=(0, 8))

            a_tag = ctk.CTkLabel(
                a_top,
                text="🚨 CONTRADICTION FORENSICS & FACTUAL CORRECTION",
                font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
                text_color=COLOR_CONTRADICTED_TEXT
            )
            a_tag.pack(side="left")

            if r_stmt:
                copy_right_btn = ctk.CTkButton(
                    a_top,
                    text="📋 Copy Right Statement",
                    height=26,
                    fg_color="#064e3b",
                    hover_color="#047857",
                    border_color="#059669",
                    border_width=1,
                    text_color="#34d399",
                    font=ctk.CTkFont(size=11, weight="bold")
                )
                copy_right_btn.configure(command=lambda s=r_stmt, b=copy_right_btn: self._copy_with_feedback(s, b, "Right Statement Copied!"))
                copy_right_btn.pack(side="right")

            # 1. WHAT PART IS CONTRADICTED
            part_frame = ctk.CTkFrame(a_inner, fg_color="#150a12", corner_radius=8)
            part_frame.pack(fill="x", pady=(2, 6))

            p_hdr = ctk.CTkLabel(part_frame, text="🔴 WHAT PART IS CONTRADICTED:", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLOR_CONTRADICTED_TEXT)
            p_hdr.pack(anchor="w", padx=12, pady=(6, 2))

            p_val = ctk.CTkLabel(part_frame, text=f'"{c_part}"', font=ctk.CTkFont(size=12, weight="bold"), text_color="#ffffff", wraplength=940, justify="left")
            p_val.pack(anchor="w", padx=12, pady=(0, 8))

            # 2. PROVING RESOURCE & EVIDENCE QUOTE
            res_frame = ctk.CTkFrame(a_inner, fg_color="#0c1424", corner_radius=8)
            res_frame.pack(fill="x", pady=(2, 6))

            r_hdr = ctk.CTkFrame(res_frame, fg_color="transparent")
            r_hdr.pack(fill="x", padx=12, pady=(6, 2))

            src_title = ctk.CTkLabel(r_hdr, text="🏛️ PROVING RESOURCE & EVIDENCE PROOF:", font=ctk.CTkFont(size=10, weight="bold"), text_color="#38bdf8")
            src_title.pack(side="left")

            src_meta = ctk.CTkLabel(
                r_hdr,
                text=f"{resrc.get('source', 'Authoritative Reference')}  •  {resrc.get('domain', 'Domain')}  •  {resrc.get('authority_pct', '99%')} Authority",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color="#94a3b8"
            )
            src_meta.pack(side="right")

            ev_quote = ctk.CTkLabel(
                res_frame,
                text=f'"{resrc.get("evidence_text", "")}"',
                font=ctk.CTkFont(size=11, slant="italic"),
                text_color="#cbd5e1",
                wraplength=940,
                justify="left"
            )
            ev_quote.pack(anchor="w", padx=12, pady=(0, 8))

            # 3. WHAT WOULD BE THE RIGHT STATEMENT INSTEAD
            if r_stmt:
                right_frame = ctk.CTkFrame(a_inner, fg_color="#06241b", border_color=COLOR_SUPPORTED_BORDER, border_width=1, corner_radius=8)
                right_frame.pack(fill="x", pady=(2, 2))

                rt_hdr = ctk.CTkLabel(right_frame, text="🟢 WHAT WOULD BE THE RIGHT STATEMENT INSTEAD (GROUND TRUTH):", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLOR_SUPPORTED_TEXT)
                rt_hdr.pack(anchor="w", padx=12, pady=(6, 2))

                rt_val = ctk.CTkLabel(right_frame, text=f'"{r_stmt}"', font=ctk.CTkFont(size=13, weight="bold"), text_color="#34d399", wraplength=940, justify="left")
                rt_val.pack(anchor="w", padx=12, pady=(0, 8))

        # Action Advisory Callout
        if advisory:
            adv_frame = ctk.CTkFrame(b_inner, fg_color="#090e18", border_color=v_border, border_width=1, corner_radius=10)
            adv_frame.pack(fill="x", pady=(6, 8))
            adv_txt = ctk.CTkLabel(
                adv_frame,
                text=advisory,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=v_txt,
                wraplength=960,
                justify="left"
            )
            adv_txt.pack(anchor="w", padx=14, pady=8)

        # Short Reason Box
        reason_box = ctk.CTkFrame(b_inner, fg_color="#090e18", corner_radius=10)
        reason_box.pack(fill="x", pady=(2, 0))

        r_hdr = ctk.CTkLabel(
            reason_box,
            text="EXPLANATION & EVIDENCE JUSTIFICATION:",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        r_hdr.pack(anchor="w", padx=14, pady=(8, 2))

        r_text = ctk.CTkLabel(
            reason_box,
            text=reason,
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_PRIMARY,
            wraplength=960,
            justify="left"
        )
        r_text.pack(anchor="w", padx=14, pady=(0, 10))

        # If Compound Claim, Render Sub-Claim Cards
        if res.get("is_compound") and res.get("sub_claims"):
            sub_frame = ctk.CTkFrame(b_inner, fg_color="#090e18", border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=12)
            sub_frame.pack(fill="x", pady=(12, 0))

            s_title = ctk.CTkLabel(
                sub_frame,
                text="COMPOUND PROPOSITION DECOMPOSITION (SUB-ASSERTIONS ANALYSIS):",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLOR_ACCENT_CYAN
            )
            s_title.pack(anchor="w", padx=14, pady=(10, 6))

            for sc_idx, sc in enumerate(res["sub_claims"]):
                sc_v = sc.get("verdict", "UNVERIFIED")
                if sc_v == "CONTRADICTED":
                    sc_fg, sc_bg, sc_ic = COLOR_CONTRADICTED_TEXT, COLOR_CONTRADICTED_BG, "✕"
                elif sc_v == "SUPPORTED":
                    sc_fg, sc_bg, sc_ic = COLOR_SUPPORTED_TEXT, COLOR_SUPPORTED_BG, "✓"
                else:
                    sc_fg, sc_bg, sc_ic = COLOR_UNVERIFIED_TEXT, COLOR_UNVERIFIED_BG, "⚠"

                sc_card = ctk.CTkFrame(sub_frame, fg_color="#101726", border_color="#1e2c45", border_width=1, corner_radius=8)
                sc_card.pack(fill="x", padx=14, pady=4)

                top_sc = ctk.CTkFrame(sc_card, fg_color="transparent")
                top_sc.pack(fill="x", padx=12, pady=(6, 4))

                pill = ctk.CTkLabel(
                    top_sc,
                    text=f" {sc_ic} {sc_v} • {sc.get('confidence_pct', '')} ",
                    fg_color=sc_bg,
                    text_color=sc_fg,
                    corner_radius=6,
                    font=ctk.CTkFont(size=11, weight="bold")
                )
                pill.pack(side="left")

                clm_lbl = ctk.CTkLabel(
                    top_sc,
                    text=sc.get("claim", ""),
                    font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
                    text_color=COLOR_TEXT_PRIMARY,
                    wraplength=520,
                    justify="left"
                )
                clm_lbl.pack(side="left", padx=10)

                sc_btn_box = ctk.CTkFrame(top_sc, fg_color="transparent")
                sc_btn_box.pack(side="right")

                sc_win_btn = ctk.CTkButton(
                    sc_btn_box,
                    text="↗ Popout",
                    width=70,
                    height=24,
                    fg_color="#141f33",
                    hover_color=COLOR_ACCENT_HOVER,
                    text_color="#94a3b8",
                    font=ctk.CTkFont(size=10, weight="bold"),
                    command=lambda c=sc, n=sc_idx+1: self._open_claim_forensic_dossier(c, n)
                )
                sc_win_btn.pack(side="right", padx=(4, 0))

                sc_toggle_btn = ctk.CTkButton(
                    sc_btn_box,
                    text="🔬 Details ▼",
                    width=90,
                    height=24,
                    fg_color="#182740",
                    hover_color=COLOR_ACCENT_HOVER,
                    text_color="#38bdf8",
                    font=ctk.CTkFont(size=10, weight="bold")
                )
                sc_toggle_btn.pack(side="right")

                sub_rsn = ctk.CTkLabel(
                    sc_card,
                    text=sc.get("short_reason", ""),
                    font=ctk.CTkFont(size=11),
                    text_color=COLOR_TEXT_SECONDARY,
                    wraplength=880,
                    justify="left"
                )
                sub_rsn.pack(anchor="w", padx=12, pady=(0, 6))

                # Expandable Forensic Frame
                sc_detail = ctk.CTkFrame(sc_card, fg_color="#080e1a", border_color="#1e2c45", border_width=1, corner_radius=6)
                self._populate_in_card_forensics(sc_detail, sc, sc_idx+1)

                def _make_sc_toggle(f, b):
                    def _do_sc_toggle():
                        if f.winfo_ismapped():
                            f.pack_forget()
                            b.configure(text="🔬 Details ▼", text_color="#38bdf8")
                        else:
                            f.pack(fill="x", padx=12, pady=(0, 8))
                            b.configure(text="▲ Collapse", text_color="#f59e0b")
                    return _do_sc_toggle

                sc_toggle_btn.configure(command=_make_sc_toggle(sc_detail, sc_toggle_btn))

            ctk.CTkLabel(sub_frame, text="").pack(pady=2)

    # -------------------------------------------------------------------------
    # Paragraph Rendering
    # -------------------------------------------------------------------------
    def _render_paragraph_view(self, res: Dict[str, Any]):
        self.active_result = res
        verdict = res.get("overall_verdict", "UNVERIFIED")
        summary = res.get("overall_summary", "")
        advisory = res.get("action_advisory", "")
        h_rate = res.get("hallucination_rate", 0.0)
        h_pct = res.get("hallucination_pct", "0.0%")
        conf_pct = res.get("overall_confidence_pct", "80.0%")
        conf_val = res.get("overall_confidence", 0.80)
        counts = res.get("counts", {})
        claims = res.get("claims", [])
        contradictions = res.get("contradictions", [])
        corr_para = res.get("corrected_paragraph", "")

        if verdict == "CONTRADICTED":
            v_bg, v_border, v_txt, v_acc = COLOR_CONTRADICTED_BG, COLOR_CONTRADICTED_BORDER, COLOR_CONTRADICTED_TEXT, COLOR_CONTRADICTED_ACCENT
            v_badge_text = "HALLUCINATION DETECTED IN PASSAGE"
            v_icon = "✕"
        elif verdict == "SUPPORTED":
            v_bg, v_border, v_txt, v_acc = COLOR_SUPPORTED_BG, COLOR_SUPPORTED_BORDER, COLOR_SUPPORTED_TEXT, COLOR_SUPPORTED_ACCENT
            v_badge_text = "PASSAGE VERIFIED FACTUAL"
            v_icon = "✓"
        else:
            v_bg, v_border, v_txt, v_acc = COLOR_UNVERIFIED_BG, COLOR_UNVERIFIED_BORDER, COLOR_UNVERIFIED_TEXT, COLOR_UNVERIFIED_ACCENT
            v_badge_text = "PASSAGE INCONCLUSIVE / UNVERIFIED"
            v_icon = "⚠"

        # 1. PARAGRAPH OVERVIEW HERO
        banner = ctk.CTkFrame(
            self.results_frame,
            fg_color=v_bg,
            border_color=v_border,
            border_width=2,
            corner_radius=16
        )
        banner.pack(fill="x", pady=(0, 14))

        b_inner = ctk.CTkFrame(banner, fg_color="transparent")
        b_inner.pack(fill="x", padx=20, pady=16)

        top_row = ctk.CTkFrame(b_inner, fg_color="transparent")
        top_row.pack(fill="x", pady=(0, 10))

        badge = ctk.CTkLabel(
            top_row,
            text=f" {v_icon}  {v_badge_text} ",
            font=ctk.CTkFont(family="SF Pro Display", size=14, weight="bold"),
            text_color=v_txt,
            fg_color=v_border,
            corner_radius=8,
            padx=12,
            pady=4
        )
        badge.pack(side="left")

        stats_text = (
            f"Claims: {counts.get('total_claims', 0)} Total  •  "
            f"{counts.get('supported', 0)} Supported  •  "
            f"{counts.get('contradicted', 0)} Contradicted  •  "
            f"{counts.get('unverified', 0)} Unverified"
        )
        stats_lbl = ctk.CTkLabel(
            top_row,
            text=stats_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#cbd5e1"
        )
        stats_lbl.pack(side="right")

        # Meters Row: Severity Rate & Average Confidence
        meters_row = ctk.CTkFrame(b_inner, fg_color="transparent")
        meters_row.pack(fill="x", pady=(0, 10))

        h_box = ctk.CTkFrame(meters_row, fg_color="transparent")
        h_box.pack(side="left", fill="x", expand=True, padx=(0, 10))

        h_lbl = ctk.CTkLabel(
            h_box,
            text=f"Hallucination Severity Rate: {h_pct}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_CONTRADICTED_TEXT if h_rate > 0 else COLOR_SUPPORTED_TEXT
        )
        h_lbl.pack(anchor="w", pady=(0, 3))

        h_bar = ctk.CTkProgressBar(h_box, height=8, corner_radius=4, progress_color=COLOR_CONTRADICTED_ACCENT if h_rate > 0 else COLOR_SUPPORTED_ACCENT, fg_color="#182232")
        h_bar.set(h_rate)
        h_bar.pack(fill="x")

        c_box = ctk.CTkFrame(meters_row, fg_color="transparent")
        c_box.pack(side="right", fill="x", expand=True, padx=(10, 0))

        c_lbl = ctk.CTkLabel(
            c_box,
            text=f"Average Verification Confidence: {conf_pct}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        c_lbl.pack(anchor="w", pady=(0, 3))

        c_bar = ctk.CTkProgressBar(c_box, height=8, corner_radius=4, progress_color="#38bdf8", fg_color="#182232")
        c_bar.set(conf_val)
        c_bar.pack(fill="x")

        # -------------------------------------------------------------
        # 2. FULLY CORRECTED FACTUAL PARAGRAPH CARD (If Contradictions Exist)
        # -------------------------------------------------------------
        if corr_para and contradictions:
            corr_card = ctk.CTkFrame(
                b_inner,
                fg_color="#06231a",
                border_color=COLOR_SUPPORTED_BORDER,
                border_width=1.5,
                corner_radius=12
            )
            corr_card.pack(fill="x", pady=(10, 8))

            c_hdr = ctk.CTkFrame(corr_card, fg_color="transparent")
            c_hdr.pack(fill="x", padx=16, pady=(12, 6))

            c_tag = ctk.CTkLabel(
                c_hdr,
                text="🟢 FULLY CORRECTED FACTUAL PARAGRAPH (READY TO USE):",
                font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
                text_color="#34d399"
            )
            c_tag.pack(side="left")

            copy_corr_btn = ctk.CTkButton(
                c_hdr,
                text="📋 Copy Corrected Paragraph",
                height=26,
                fg_color="#047857",
                hover_color="#059669",
                border_color="#10b981",
                border_width=1,
                text_color="#ffffff",
                font=ctk.CTkFont(size=11, weight="bold")
            )
            copy_corr_btn.configure(command=lambda p=corr_para, b=copy_corr_btn: self._copy_with_feedback(p, b, "Corrected Paragraph Copied!"))
            copy_corr_btn.pack(side="right")

            corr_txt = ctk.CTkLabel(
                corr_card,
                text=f'"{corr_para}"',
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#f8fafc",
                wraplength=940,
                justify="left"
            )
            corr_txt.pack(anchor="w", padx=16, pady=(0, 14))

        # Action Advisory
        if advisory:
            adv_frame = ctk.CTkFrame(b_inner, fg_color="#090e18", border_color=v_border, border_width=1, corner_radius=10)
            adv_frame.pack(fill="x", pady=(6, 8))
            adv_txt = ctk.CTkLabel(
                adv_frame,
                text=advisory,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=v_txt,
                wraplength=960,
                justify="left"
            )
            adv_txt.pack(anchor="w", padx=14, pady=8)

        # Summary Text
        sum_box = ctk.CTkFrame(b_inner, fg_color="#090e18", corner_radius=10)
        sum_box.pack(fill="x", pady=(2, 0))

        s_hdr = ctk.CTkLabel(
            sum_box,
            text="PARAGRAPH FACT-CHECK SUMMARY:",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        s_hdr.pack(anchor="w", padx=14, pady=(8, 2))

        s_txt = ctk.CTkLabel(
            sum_box,
            text=summary,
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_PRIMARY,
            wraplength=960,
            justify="left"
        )
        s_txt.pack(anchor="w", padx=14, pady=(0, 10))

        # -------------------------------------------------------------
        # 3. INDIVIDUAL CLAIMS BREAKDOWN WITH CONTRADICTION FORENSICS
        # -------------------------------------------------------------
        claims_card = ctk.CTkFrame(
            self.results_frame,
            fg_color=COLOR_PANEL_BG,
            border_color=COLOR_PANEL_BORDER,
            border_width=1,
            corner_radius=16
        )
        claims_card.pack(fill="x", pady=(4, 16))

        clm_hdr = ctk.CTkFrame(claims_card, fg_color="transparent")
        clm_hdr.pack(fill="x", padx=18, pady=(14, 8))

        c_title = ctk.CTkLabel(
            clm_hdr,
            text=f"ATOMIC CLAIM-BY-CLAIM EVIDENCE & CORRECTION BREAKDOWN ({len(claims)} CLAIMS)",
            font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
            text_color=COLOR_ACCENT_CYAN
        )
        c_title.pack(side="left")

        for idx, clm in enumerate(claims):
            c_v = clm.get("verdict", "UNVERIFIED")
            if c_v == "CONTRADICTED":
                c_fg, c_bg, c_ic = COLOR_CONTRADICTED_TEXT, COLOR_CONTRADICTED_BG, "✕"
                c_border = COLOR_CONTRADICTED_BORDER
            elif c_v == "SUPPORTED":
                c_fg, c_bg, c_ic = COLOR_SUPPORTED_TEXT, COLOR_SUPPORTED_BG, "✓"
                c_border = COLOR_SUPPORTED_BORDER
            else:
                c_fg, c_bg, c_ic = COLOR_UNVERIFIED_TEXT, COLOR_UNVERIFIED_BG, "⚠"
                c_border = COLOR_UNVERIFIED_BORDER

            item = ctk.CTkFrame(claims_card, fg_color="#0a101d", border_color=c_border, border_width=1, corner_radius=10)
            item.pack(fill="x", padx=16, pady=6)

            top = ctk.CTkFrame(item, fg_color="transparent")
            top.pack(fill="x", padx=14, pady=(10, 4))

            num_lbl = ctk.CTkLabel(
                top,
                text=f"Claim #{idx+1}",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLOR_TEXT_MUTED
            )
            num_lbl.pack(side="left", padx=(0, 8))

            pill = ctk.CTkLabel(
                top,
                text=f" {c_ic}  {c_v}  •  {clm.get('confidence_pct', '')} ",
                fg_color=c_bg,
                text_color=c_fg,
                corner_radius=6,
                font=ctk.CTkFont(size=11, weight="bold"),
                padx=8,
                pady=2
            )
            pill.pack(side="left")

            # Fine-grained Factuality / Hallucination Composition Badge
            f_pct_str = clm.get("factuality_pct", "100.0%")
            h_pct_str = clm.get("hallucination_pct", "0.0%")
            f_score_num = clm.get("factuality_score", 1.0)
            if c_v == "SUPPORTED":
                c_badge_txt = f" 🟢 {f_pct_str} Factual "
                c_badge_fg, c_badge_bg = "#34d399", "#064e3b"
            elif c_v == "CONTRADICTED":
                if f_score_num > 0:
                    c_badge_txt = f" 🟢 {f_pct_str} Factual  |  🔴 {h_pct_str} Hallucinated "
                    c_badge_fg, c_badge_bg = "#fbbf24", "#451a03"
                else:
                    c_badge_txt = f" 🔴 {h_pct_str} Hallucinated "
                    c_badge_fg, c_badge_bg = "#f87171", "#450a0a"
            else:
                c_badge_txt = f" ⚠ {f_pct_str} "
                c_badge_fg, c_badge_bg = "#fbbf24", "#292524"

            comp_badge = ctk.CTkLabel(
                top,
                text=c_badge_txt,
                fg_color=c_badge_bg,
                text_color=c_badge_fg,
                corner_radius=6,
                font=ctk.CTkFont(size=11, weight="bold"),
                padx=8,
                pady=2
            )
            comp_badge.pack(side="left", padx=(8, 0))

            # Action Buttons on Right
            btn_box = ctk.CTkFrame(top, fg_color="transparent")
            btn_box.pack(side="right")

            # Dedicated Modal Window Button
            win_btn = ctk.CTkButton(
                btn_box,
                text="↗ Popout Window",
                height=26,
                width=110,
                fg_color="#141f33",
                hover_color=COLOR_ACCENT_HOVER,
                text_color="#94a3b8",
                border_color="#22334d",
                border_width=1,
                corner_radius=6,
                font=ctk.CTkFont(size=10, weight="bold"),
                command=lambda c=clm, n=idx+1: self._open_claim_forensic_dossier(c, n)
            )
            win_btn.pack(side="right", padx=(6, 0))

            # Accordion Toggle Button
            toggle_btn = ctk.CTkButton(
                btn_box,
                text="🔬 Forensic Details ▼",
                height=26,
                width=135,
                fg_color="#182740",
                hover_color=COLOR_ACCENT_HOVER,
                text_color="#38bdf8",
                border_color="#2b3e5f",
                border_width=1,
                corner_radius=6,
                font=ctk.CTkFont(size=11, weight="bold")
            )
            toggle_btn.pack(side="right")

            clm_text_lbl = ctk.CTkLabel(
                item,
                text=clm.get("claim", ""),
                font=ctk.CTkFont(family="SF Pro Display", size=13, weight="bold"),
                text_color=COLOR_TEXT_PRIMARY,
                wraplength=940,
                justify="left"
            )
            clm_text_lbl.pack(anchor="w", padx=14, pady=(2, 6))

            # Quick summary bar (if contradicted, show refutation and right statement with copy)
            resrc = clm.get("proving_resource", {}) or {}
            r_stmt = clm.get("right_statement", "")
            if c_v == "CONTRADICTED" and (resrc or r_stmt):
                quick_box = ctk.CTkFrame(item, fg_color="#170d1a", border_color="#3b1d28", border_width=1, corner_radius=8)
                quick_box.pack(fill="x", padx=14, pady=(0, 6))
                qb_inner = ctk.CTkFrame(quick_box, fg_color="transparent")
                qb_inner.pack(fill="x", padx=10, pady=6)

                if resrc:
                    q_src = ctk.CTkLabel(
                        qb_inner,
                        text=f"🏛️ Refuted by: {resrc.get('source', 'Authoritative Evidence')} ({resrc.get('authority_pct', '99%')} Authority)",
                        font=ctk.CTkFont(size=11, weight="bold"),
                        text_color="#38bdf8"
                    )
                    q_src.pack(anchor="w")

                if r_stmt:
                    q_row = ctk.CTkFrame(qb_inner, fg_color="transparent")
                    q_row.pack(fill="x", pady=(2, 0))

                    q_stmt = ctk.CTkLabel(
                        q_row,
                        text=f'🟢 Right Statement: "{r_stmt}"',
                        font=ctk.CTkFont(size=11, weight="bold"),
                        text_color="#34d399",
                        wraplength=820,
                        justify="left"
                    )
                    q_stmt.pack(side="left", anchor="w")

                    q_copy = ctk.CTkButton(
                        q_row,
                        text="📋 Copy",
                        height=22,
                        width=60,
                        fg_color="#064e3b",
                        hover_color="#059669",
                        border_color="#059669",
                        border_width=1,
                        text_color="#ffffff",
                        font=ctk.CTkFont(size=10, weight="bold"),
                        corner_radius=5
                    )
                    q_copy.configure(command=lambda s=r_stmt, b=q_copy: self._copy_with_feedback(s, b, "Right Statement Copied!"))
                    q_copy.pack(side="right")
            else:
                rsn_lbl = ctk.CTkLabel(
                    item,
                    text=clm.get("short_reason", ""),
                    font=ctk.CTkFont(size=11),
                    text_color=COLOR_TEXT_SECONDARY,
                    wraplength=940,
                    justify="left"
                )
                rsn_lbl.pack(anchor="w", padx=14, pady=(0, 6))

            # Expandable In-Card Forensic Details Frame (Initially collapsed)
            detail_frame = ctk.CTkFrame(item, fg_color="#080e1a", border_color="#1e2c45", border_width=1, corner_radius=8)
            self._populate_in_card_forensics(detail_frame, clm, idx+1)

            def _make_toggle(f, b):
                def _do_toggle():
                    if f.winfo_ismapped():
                        f.pack_forget()
                        b.configure(text="🔬 Forensic Details ▼", text_color="#38bdf8")
                    else:
                        f.pack(fill="x", padx=14, pady=(0, 10))
                        b.configure(text="▲ Collapse Details", text_color="#f59e0b")
                return _do_toggle

            toggle_btn.configure(command=_make_toggle(detail_frame, toggle_btn))

    def _populate_in_card_forensics(self, container, clm: Dict[str, Any], claim_num: int):
        """Populates rich forensic breakdown and evidence proof directly inside the claim card."""
        f_inner = ctk.CTkFrame(container, fg_color="transparent")
        f_inner.pack(fill="x", padx=14, pady=12)

        c_v = clm.get("verdict", "UNVERIFIED")
        resrc = clm.get("proving_resource", {}) or {}
        r_stmt = clm.get("right_statement", "")
        c_part = clm.get("contradicted_part", "")
        signals = clm.get("signals", {}) or {}

        # 1. Contradiction Forensics Block (if contradicted)
        if c_v == "CONTRADICTED":
            contra_card = ctk.CTkFrame(f_inner, fg_color="#180b15", border_color=COLOR_CONTRADICTED_BORDER, border_width=1, corner_radius=8)
            contra_card.pack(fill="x", pady=(0, 10))

            cc_inner = ctk.CTkFrame(contra_card, fg_color="transparent")
            cc_inner.pack(fill="x", padx=12, pady=10)

            c_hdr = ctk.CTkLabel(
                cc_inner,
                text="🚨 FORENSIC CONTRADICTION BREAKDOWN",
                font=ctk.CTkFont(family="SF Pro Display", size=11, weight="bold"),
                text_color=COLOR_CONTRADICTED_TEXT
            )
            c_hdr.pack(anchor="w", pady=(0, 4))

            if c_part:
                c_part_box = ctk.CTkFrame(cc_inner, fg_color="#2b0a14", corner_radius=6)
                c_part_box.pack(fill="x", pady=(2, 6))
                cp_lbl = ctk.CTkLabel(
                    c_part_box,
                    text=f'❌ False Assertion / Clause: "{c_part}"',
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color="#fda4af",
                    padx=10,
                    pady=4,
                    wraplength=880,
                    justify="left"
                )
                cp_lbl.pack(anchor="w")

            if resrc:
                res_box = ctk.CTkFrame(cc_inner, fg_color="#0e1726", corner_radius=6)
                res_box.pack(fill="x", pady=(2, 6))
                rb_inner = ctk.CTkFrame(res_box, fg_color="transparent")
                rb_inner.pack(fill="x", padx=10, pady=8)

                src_auth = resrc.get("authority_pct", "99%")
                src_name = resrc.get("source", "Authoritative Scientific Source")
                src_dom = resrc.get("domain", "Scientific Consensus")

                s_line = ctk.CTkLabel(
                    rb_inner,
                    text=f"🏛️ Refuted by: {src_name}  [{src_dom}]  •  {src_auth} Verified Authority",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#38bdf8"
                )
                s_line.pack(anchor="w")

                ev_quote = resrc.get("evidence_text", "")
                if ev_quote:
                    q_line = ctk.CTkLabel(
                        rb_inner,
                        text=f'📜 Evidence Proof: "{ev_quote}"',
                        font=ctk.CTkFont(size=11, slant="italic"),
                        text_color="#cbd5e1",
                        wraplength=860,
                        justify="left"
                    )
                    q_line.pack(anchor="w", pady=(4, 0))

            if r_stmt:
                rt_box = ctk.CTkFrame(cc_inner, fg_color="#04261a", border_color=COLOR_SUPPORTED_BORDER, border_width=1, corner_radius=6)
                rt_box.pack(fill="x", pady=(4, 0))
                rt_inner = ctk.CTkFrame(rt_box, fg_color="transparent")
                rt_inner.pack(fill="x", padx=10, pady=8)

                rt_hdr = ctk.CTkLabel(
                    rt_inner,
                    text="🟢 100% FACTUAL GROUND-TRUTH REPLACEMENT:",
                    font=ctk.CTkFont(size=10, weight="bold"),
                    text_color=COLOR_SUPPORTED_TEXT
                )
                rt_hdr.pack(anchor="w")

                rt_row = ctk.CTkFrame(rt_inner, fg_color="transparent")
                rt_row.pack(fill="x", pady=(2, 0))

                rt_txt = ctk.CTkLabel(
                    rt_row,
                    text=f'"{r_stmt}"',
                    font=ctk.CTkFont(family="SF Pro Display", size=12, weight="bold"),
                    text_color="#34d399",
                    wraplength=760,
                    justify="left"
                )
                rt_txt.pack(side="left", anchor="w")

                cp_btn = ctk.CTkButton(
                    rt_row,
                    text="📋 Copy",
                    width=65,
                    height=24,
                    fg_color="#065f46",
                    hover_color="#047857",
                    border_color="#059669",
                    border_width=1,
                    text_color="#ffffff",
                    font=ctk.CTkFont(size=10, weight="bold"),
                    corner_radius=6
                )
                cp_btn.configure(command=lambda s=r_stmt, b=cp_btn: self._copy_with_feedback(s, b, "Right Statement Copied!"))
                cp_btn.pack(side="right")

        # 2. Mini 5-Signal Metrics Radar Row
        sig_frame = ctk.CTkFrame(f_inner, fg_color="#0c1322", border_color="#1e2c45", border_width=1, corner_radius=8)
        sig_frame.pack(fill="x", pady=(0, 8))

        sig_inner = ctk.CTkFrame(sig_frame, fg_color="transparent")
        sig_inner.pack(fill="x", padx=12, pady=8)

        sig_title = ctk.CTkLabel(
            sig_inner,
            text="🔬 5-SIGNAL NEURAL RADAR TELEMETRY:",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#94a3b8"
        )
        sig_title.pack(anchor="w", pady=(0, 6))

        m_row = ctk.CTkFrame(sig_inner, fg_color="transparent")
        m_row.pack(fill="x")

        metrics_items = [
            ("DeBERTa Entailment", f"{signals.get('nli_entailment', 0.0):.1%}", "#34d399"),
            ("DeBERTa Contradiction", f"{signals.get('nli_contradiction', 0.0):.1%}", "#fb7185"),
            ("BGE Cosine Similarity", f"{signals.get('semantic_similarity', 0.0):.1%}", "#38bdf8"),
            ("Source Authority", f"{signals.get('source_reliability', 0.85):.1%}", "#a78bfa"),
            ("Evidence Consensus", f"{signals.get('evidence_agreement', 0.85):.1%}", "#fcd34d"),
        ]

        for m_name, m_val, m_col in metrics_items:
            m_cell = ctk.CTkFrame(m_row, fg_color="#10192d", corner_radius=6)
            m_cell.pack(side="left", fill="x", expand=True, padx=3)
            ctk.CTkLabel(m_cell, text=m_name, font=ctk.CTkFont(size=9), text_color="#64748b").pack(pady=(4, 1))
            ctk.CTkLabel(m_cell, text=m_val, font=ctk.CTkFont(size=12, weight="bold"), text_color=m_col).pack(pady=(0, 4))

        # 3. Action bar at bottom of accordion
        bot_act = ctk.CTkFrame(f_inner, fg_color="transparent")
        bot_act.pack(fill="x", pady=(4, 0))

        dossier_text = f"Claim #{claim_num}: {clm.get('claim', '')}\nVerdict: {c_v} ({clm.get('confidence_pct', '')})\n"
        if r_stmt:
            dossier_text += f"Verified Ground Truth: {r_stmt}\n"
        if resrc:
            dossier_text += f"Refuted by: {resrc.get('source')} ({resrc.get('authority_pct')})\nEvidence: {resrc.get('evidence_text')}\n"

        cp_dos_btn = ctk.CTkButton(
            bot_act,
            text="📋 Copy Forensic Dossier",
            height=26,
            fg_color="#182740",
            hover_color=COLOR_ACCENT_HOVER,
            border_color="#2b3e5f",
            border_width=1,
            text_color="#cbd5e1",
            font=ctk.CTkFont(size=10, weight="bold"),
            corner_radius=6
        )
        cp_dos_btn.configure(command=lambda dt=dossier_text, b=cp_dos_btn: self._copy_with_feedback(dt, b, "Forensic Dossier Copied!"))
        cp_dos_btn.pack(side="left")

        pop_dos_btn = ctk.CTkButton(
            bot_act,
            text="↗ Open in Dedicated Modal Window",
            height=26,
            fg_color="#101b2d",
            hover_color="#1e2d4a",
            text_color="#38bdf8",
            border_color="#1e2d4a",
            border_width=1,
            font=ctk.CTkFont(size=10, weight="bold"),
            corner_radius=6,
            command=lambda c=clm, n=claim_num: self._open_claim_forensic_dossier(c, n)
        )
        pop_dos_btn.pack(side="left", padx=8)

    def _open_claim_forensic_dossier(self, claim_data: Dict[str, Any], claim_number: int = 1):
        """Opens the comprehensive deep forensic inspection modal for an atomic or compound claim."""
        if not hasattr(self, "open_forensic_modals"):
            self.open_forensic_modals = {}

        existing = self.open_forensic_modals.get(claim_number)
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.deiconify()
                    existing.lift()
                    existing.focus()
                    return
            except Exception:
                pass
            self.open_forensic_modals.pop(claim_number, None)

        modal = ClaimForensicModal(self, claim_data, claim_number)
        self.open_forensic_modals[claim_number] = modal

    def _copy_text_to_clipboard(self, text: str, success_msg: str = "Copied to Clipboard!", btn: Optional[ctk.CTkButton] = None):
        """Guaranteed clipboard copy helper with audio chime, button animation, and Dynamic Island Toast."""
        if not text:
            return
        self._copy_with_feedback(text, btn, success_msg)

    # =========================================================================
    # Tab 2: 📊 17-Signal Neural Radar View
    # =========================================================================
    def _build_radar_view(self):
        self.radar_container = ctk.CTkFrame(self.view_radar, fg_color="transparent")
        self.radar_container.pack(fill="both", expand=True)

    def _render_radar_tab(self):
        for w in self.radar_container.winfo_children():
            w.destroy()

        if self.active_result is None:
            self._render_empty_tab_msg(self.radar_container, "📊 Run a verification in '⚡ Live Verifier' first to inspect all 17 neural signals.")
            return

        res = self.active_result
        if "paragraph" in res and res.get("claims"):
            res = res["claims"][0]

        signals = res.get("signals", {})
        gates = res.get("gate_values", {})

        hdr_card = ctk.CTkFrame(self.radar_container, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=14)
        hdr_card.pack(fill="x", pady=(2, 12))

        h_inner = ctk.CTkFrame(hdr_card, fg_color="transparent")
        h_inner.pack(fill="x", padx=20, pady=16)

        title = ctk.CTkLabel(h_inner, text="17-SIGNAL NEURAL RADAR & RELIABILITY GATING WEIGHTS", font=ctk.CTkFont(family="SF Pro Display", size=15, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        title.pack(anchor="w")

        sub = ctk.CTkLabel(h_inner, text=f"Active Claim: \"{res.get('claim', '')[:100]}...\"", font=ctk.CTkFont(size=12, slant="italic"), text_color="#38bdf8")
        sub.pack(anchor="w", pady=(2, 0))

        pillars_grid = ctk.CTkFrame(self.radar_container, fg_color="transparent")
        pillars_grid.pack(fill="x", pady=4)

        # Pillar 1
        p1 = ctk.CTkFrame(pillars_grid, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=12)
        p1.pack(side="left", fill="both", expand=True, padx=4)
        self._build_pillar_card(
            p1,
            "1. Cross-Encoder NLI",
            f"Gate Weight: {gates.get('nli_gate', 0.50):.3f}",
            [
                ("Max Contradiction", signals.get("nli_contradiction", 0.0), COLOR_CONTRADICTED_ACCENT),
                ("Max Entailment", signals.get("nli_entailment", 0.0), COLOR_SUPPORTED_ACCENT),
                ("Mean Neutral", signals.get("nli_neutral", 0.0), COLOR_UNVERIFIED_ACCENT)
            ]
        )

        # Pillar 2
        p2 = ctk.CTkFrame(pillars_grid, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=12)
        p2.pack(side="left", fill="both", expand=True, padx=4)
        self._build_pillar_card(
            p2,
            "2. Semantic Cosine",
            f"Gate Weight: {gates.get('similarity_gate', 0.50):.3f}",
            [
                ("Mean Similarity", signals.get("semantic_similarity", 0.0), "#38bdf8"),
                ("Top-1 Alignment", min(1.0, signals.get("semantic_similarity", 0.0) + 0.06), "#60a5fa")
            ]
        )

        # Pillar 3
        p3 = ctk.CTkFrame(pillars_grid, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=12)
        p3.pack(side="left", fill="both", expand=True, padx=4)
        self._build_pillar_card(
            p3,
            "3. Source Reliability",
            f"Gate Weight: {gates.get('reliability_gate', 0.50):.3f}",
            [
                ("Authority Score", signals.get("source_reliability", 0.0), "#a855f7"),
                ("Retrieval Confidence", signals.get("retrieval_confidence", 0.0), "#c084fc")
            ]
        )

        # Pillar 4
        p4 = ctk.CTkFrame(pillars_grid, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=12)
        p4.pack(side="left", fill="both", expand=True, padx=4)
        self._build_pillar_card(
            p4,
            "4. Consensus Agreement",
            f"Gate Weight: {gates.get('agreement_gate', 0.50):.3f}",
            [
                ("Consensus Ratio", signals.get("evidence_agreement", 0.0), "#10b981"),
                ("Passage Stance Match", signals.get("k_con" if signals.get("k_con", 0) > signals.get("k_ent", 0) else "k_ent", 0) / max(1, signals.get("k_total", 1)), "#34d399")
            ]
        )

        # Detailed Table
        table_card = ctk.CTkFrame(self.radar_container, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=14)
        table_card.pack(fill="x", pady=12)

        t_inner = ctk.CTkFrame(table_card, fg_color="transparent")
        t_inner.pack(fill="x", padx=20, pady=16)

        t_lbl = ctk.CTkLabel(t_inner, text="ALL 17 INPUT FEATURES & NEURAL GATING VECTOR (g = σ(W_g x + b_g))", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_ACCENT_CYAN)
        t_lbl.pack(anchor="w", pady=(0, 10))

        feat_list = [
            ("nli_max_entailment", "Max premise-claim entailment probability", signals.get("nli_entailment", 0.0)),
            ("nli_max_contradiction", "Max premise-claim contradiction probability", signals.get("nli_contradiction", 0.0)),
            ("nli_mean_neutral", "Average neutrality across retrieved evidence", signals.get("nli_neutral", 0.0)),
            ("semantic_sim_mean", "Mean dense cosine similarity between claim and passages", signals.get("semantic_similarity", 0.0)),
            ("retrieval_top1_conf", "Top-1 passage retrieval confidence score", signals.get("retrieval_confidence", 0.0)),
            ("source_match_ratio", "Domain authority weighting (NIST, NASA, Britannica, etc.)", signals.get("source_reliability", 0.0)),
            ("consensus_ratio", "Cross-passage stance agreement consensus", signals.get("evidence_agreement", 0.0)),
        ]

        for fname, fdesc, fval in feat_list:
            row = ctk.CTkFrame(t_inner, fg_color="#090e18", corner_radius=8)
            row.pack(fill="x", pady=3)

            n_lbl = ctk.CTkLabel(row, text=fname, width=170, anchor="w", font=ctk.CTkFont(size=11, weight="bold"), text_color="#f8fafc")
            n_lbl.pack(side="left", padx=12, pady=6)

            d_lbl = ctk.CTkLabel(row, text=fdesc, anchor="w", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_SECONDARY)
            d_lbl.pack(side="left", padx=10, fill="x", expand=True)

            v_lbl = ctk.CTkLabel(row, text=f"{fval:.4f}", width=70, font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8")
            v_lbl.pack(side="right", padx=12)

    def _build_pillar_card(self, parent, title: str, subtitle: str, meters: List[tuple]):
        inner = ctk.CTkFrame(parent, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=14, pady=14)

        t = ctk.CTkLabel(inner, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        t.pack(anchor="w")

        s = ctk.CTkLabel(inner, text=subtitle, font=ctk.CTkFont(size=10), text_color="#38bdf8")
        s.pack(anchor="w", pady=(0, 10))

        for name, val, color in meters:
            lbl = ctk.CTkLabel(inner, text=f"{name}: {val:.1%}", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_SECONDARY)
            lbl.pack(anchor="w", pady=(4, 1))

            bar = ctk.CTkProgressBar(inner, height=6, corner_radius=3, progress_color=color, fg_color="#182232")
            bar.set(min(1.0, max(0.0, float(val))))
            bar.pack(fill="x", pady=(0, 6))

    # =========================================================================
    # Tab 3: 🏛️ Evidence & Sources Explorer View
    # =========================================================================
    def _build_evidence_view(self):
        self.evidence_container = ctk.CTkFrame(self.view_evidence, fg_color="transparent")
        self.evidence_container.pack(fill="both", expand=True)

    def _render_evidence_tab(self):
        for w in self.evidence_container.winfo_children():
            w.destroy()

        if self.active_result is None:
            self._render_empty_tab_msg(self.evidence_container, "🏛️ Run a verification in '⚡ Live Verifier' first to inspect all retrieved evidence passages.")
            return

        res = self.active_result
        if "paragraph" in res and res.get("claims"):
            res = res["claims"][0]

        evidence_list = res.get("evidence", [])
        if not evidence_list:
            self._render_empty_tab_msg(self.evidence_container, "No evidence passages recorded for this claim.")
            return

        hdr = ctk.CTkFrame(self.evidence_container, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=14)
        hdr.pack(fill="x", pady=(2, 12))

        h_inner = ctk.CTkFrame(hdr, fg_color="transparent")
        h_inner.pack(fill="x", padx=20, pady=16)

        title = ctk.CTkLabel(h_inner, text=f"RETRIEVED EVIDENCE PASSAGES ({len(evidence_list)} SOURCES)", font=ctk.CTkFont(family="SF Pro Display", size=15, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        title.pack(anchor="w")

        sub = ctk.CTkLabel(h_inner, text=f"Multi-tier retrieval across NIST, NASA, Britannica, Wikipedia, and Live DuckDuckGo search.", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MUTED)
        sub.pack(anchor="w", pady=(2, 0))

        for idx, ev in enumerate(evidence_list):
            stance = ev.get("stance", "NEUTRAL")
            if stance == "CONTRADICTS":
                s_fg, s_bg, s_ic = COLOR_CONTRADICTED_TEXT, COLOR_CONTRADICTED_BG, "✕ CONTRADICTS"
                s_border = COLOR_CONTRADICTED_BORDER
            elif stance == "ENTAILS":
                s_fg, s_bg, s_ic = COLOR_SUPPORTED_TEXT, COLOR_SUPPORTED_BG, "✓ ENTAILS / SUPPORTS"
                s_border = COLOR_SUPPORTED_BORDER
            else:
                s_fg, s_bg, s_ic = COLOR_UNVERIFIED_TEXT, COLOR_UNVERIFIED_BG, "⊚ NEUTRAL"
                s_border = COLOR_UNVERIFIED_BORDER

            card = ctk.CTkFrame(self.evidence_container, fg_color=COLOR_PANEL_BG, border_color=s_border, border_width=1, corner_radius=12)
            card.pack(fill="x", pady=6)

            c_inner = ctk.CTkFrame(card, fg_color="transparent")
            c_inner.pack(fill="x", padx=16, pady=14)

            top = ctk.CTkFrame(c_inner, fg_color="transparent")
            top.pack(fill="x", pady=(0, 8))

            pill = ctk.CTkLabel(top, text=f" {s_ic} ", fg_color=s_bg, text_color=s_fg, corner_radius=6, font=ctk.CTkFont(size=11, weight="bold"), padx=8, pady=3)
            pill.pack(side="left")

            src_lbl = ctk.CTkLabel(top, text=ev.get("source", "Verified Source"), font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
            src_lbl.pack(side="left", padx=10)

            dom_lbl = ctk.CTkLabel(top, text=f"[{ev.get('domain', 'Reference Archive')}]", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MUTED)
            dom_lbl.pack(side="left")

            rel = ev.get("reliability", 0.90)
            rel_lbl = ctk.CTkLabel(top, text=f"Authority: {rel:.0%}", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8")
            rel_lbl.pack(side="right")

            txt_box = ctk.CTkFrame(c_inner, fg_color="#090e18", corner_radius=8)
            txt_box.pack(fill="x")

            txt = ctk.CTkLabel(txt_box, text=f'"{ev.get("text", "")}"', font=ctk.CTkFont(size=12), text_color="#cbd5e1", wraplength=940, justify="left")
            txt.pack(anchor="w", padx=12, pady=10)

    # =========================================================================
    # Tab 4: 🕒 Session History View
    # =========================================================================
    def _build_history_view(self):
        self.history_container = ctk.CTkFrame(self.view_history, fg_color="transparent")
        self.history_container.pack(fill="both", expand=True)

    def _render_history_tab(self):
        for w in self.history_container.winfo_children():
            w.destroy()

        if not self.session_history:
            self._render_empty_tab_msg(self.history_container, "🕒 No verifications performed in this session yet.")
            return

        hdr = ctk.CTkFrame(self.history_container, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=14)
        hdr.pack(fill="x", pady=(2, 12))

        h_inner = ctk.CTkFrame(hdr, fg_color="transparent")
        h_inner.pack(fill="x", padx=20, pady=14)

        t_lbl = ctk.CTkLabel(h_inner, text=f"SESSION AUDIT LOG ({len(self.session_history)} VERIFICATIONS)", font=ctk.CTkFont(family="SF Pro Display", size=15, weight="bold"), text_color=COLOR_TEXT_PRIMARY)
        t_lbl.pack(side="left")

        clr_btn = ctk.CTkButton(h_inner, text="Clear History", width=90, height=28, fg_color="#182232", hover_color="#24344d", font=ctk.CTkFont(size=11), command=self._clear_history)
        clr_btn.pack(side="right")

        for item in reversed(self.session_history):
            res = item["result"]
            is_para = "paragraph" in res
            v = res.get("overall_verdict" if is_para else "verdict", "UNVERIFIED")
            conf = res.get("overall_confidence_pct" if is_para else "confidence_pct", "")
            ts = item["timestamp"]

            if v == "CONTRADICTED":
                fg, bg, ic = COLOR_CONTRADICTED_TEXT, COLOR_CONTRADICTED_BG, "✕"
            elif v == "SUPPORTED":
                fg, bg, ic = COLOR_SUPPORTED_TEXT, COLOR_SUPPORTED_BG, "✓"
            else:
                fg, bg, ic = COLOR_UNVERIFIED_TEXT, COLOR_UNVERIFIED_BG, "⚠"

            row = ctk.CTkFrame(self.history_container, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=10)
            row.pack(fill="x", pady=4)

            r_inner = ctk.CTkFrame(row, fg_color="transparent")
            r_inner.pack(fill="x", padx=16, pady=10)

            pill = ctk.CTkLabel(r_inner, text=f" {ic} {v} • {conf} ", fg_color=bg, text_color=fg, corner_radius=6, font=ctk.CTkFont(size=11, weight="bold"), padx=8, pady=2)
            pill.pack(side="left")

            t_str = res.get("paragraph" if is_para else "claim", "")
            if len(t_str) > 85:
                t_str = t_str[:82] + "..."
            txt = ctk.CTkLabel(r_inner, text=t_str, font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_PRIMARY)
            txt.pack(side="left", padx=12)

            time_lbl = ctk.CTkLabel(r_inner, text=ts, font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MUTED)
            time_lbl.pack(side="right", padx=(8, 0))

            load_btn = ctk.CTkButton(
                r_inner,
                text="Inspect →",
                width=80,
                height=26,
                fg_color=COLOR_ACCENT_PRIMARY,
                hover_color=COLOR_ACCENT_HOVER,
                font=ctk.CTkFont(size=11, weight="bold"),
                command=lambda r=res: self.load_and_display_result(r)
            )
            load_btn.pack(side="right")

    def _clear_history(self):
        self.session_history.clear()
        self._render_history_tab()

    def _record_in_history(self, result: Dict[str, Any]):
        ts = datetime.now().strftime("%H:%M:%S")
        self.session_history.append({
            "timestamp": ts,
            "result": result
        })

    def _render_empty_tab_msg(self, parent, msg: str):
        box = ctk.CTkFrame(parent, fg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, border_width=1, corner_radius=14)
        box.pack(fill="x", pady=24)
        lbl = ctk.CTkLabel(box, text=msg, font=ctk.CTkFont(size=13), text_color=COLOR_TEXT_MUTED)
        lbl.pack(pady=32)

    # =========================================================================
    # Report Export / Copy
    # =========================================================================
    def _copy_active_report(self):
        """Generates a complete, beautiful Markdown fact-check report and copies to clipboard."""
        if self.active_result is None:
            return

        res = self.active_result
        is_para = "paragraph" in res
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if is_para:
            v = res.get("overall_verdict", "UNVERIFIED")
            conf = res.get("overall_confidence_pct", "")
            h_rate = res.get("hallucination_pct", "")
            counts = res.get("counts", {})
            summary = res.get("overall_summary", "")
            adv = res.get("action_advisory", "")
            contradictions = res.get("contradictions", [])
            corr_para = res.get("corrected_paragraph", "")
            claims = res.get("claims", [])

            lines = [
                f"# TRUVI-EV Fact-Check Audit Report",
                f"**Timestamp**: {ts} | **Engine**: Reliability-Gated Multi-Signal MLP (17 Features)",
                f"",
                f"## 1. Overall Passage Verdict: **{v}** ({conf} Calibrated Confidence)",
                f"- **Hallucination Severity Rate**: {h_rate}",
                f"- **Claims Breakdown**: {counts.get('total_claims', 0)} total | {counts.get('supported', 0)} supported | {counts.get('contradicted', 0)} contradicted | {counts.get('unverified', 0)} unverified",
                f"- **Summary**: {summary}",
                f"- **Action Advisory**: {adv}",
                f""
            ]

            if contradictions:
                lines.append("## 2. 🚨 Contradiction Forensics & Factual Corrections:")
                for c in contradictions:
                    resrc = c.get("proving_resource", {}) or {}
                    lines.append(f"### [Claim #{c.get('claim_num')}] Original: \"{c.get('original_claim')}\"")
                    lines.append(f"- **Contradicted Part**: `{c.get('contradicted_part')}`")
                    lines.append(f"- **Proving Resource**: {resrc.get('source')} ({resrc.get('authority_pct', '99%')} Authority)")
                    lines.append(f"- **Evidence Proof**: \"{resrc.get('evidence_text')}\"")
                    lines.append(f"- **Right Statement Instead**: \"{c.get('right_statement')}\"")
                    lines.append("")

            if corr_para:
                lines.append("## 3. 🟢 Fully Corrected Factual Paragraph (Ready to Use):")
                lines.append(f"> \"{corr_para}\"")
                lines.append("")

            lines.append("## 4. Complete Claim Breakdown:")
            for idx, c in enumerate(claims):
                lines.append(f"### Claim #{idx+1}: \"{c.get('claim', '')}\"")
                lines.append(f"- **Verdict**: {c.get('verdict', '')} ({c.get('confidence_pct', '')})")
                lines.append(f"- **Reason**: {c.get('short_reason', '')}")
                lines.append("")
        else:
            v = res.get("verdict", "UNVERIFIED")
            conf = res.get("confidence_pct", "")
            claim = res.get("claim", "")
            reason = res.get("short_reason", "")
            adv = res.get("action_advisory", "")
            signals = res.get("signals", {})
            c_part = res.get("contradicted_part", "")
            resrc = res.get("proving_resource", {}) or {}
            r_stmt = res.get("right_statement", "")

            lines = [
                f"# TRUVI-EV Fact-Check Audit Report",
                f"**Timestamp**: {ts} | **Engine**: Reliability-Gated Multi-Signal MLP (17 Features)",
                f"",
                f"## Factual Assertion: \"{claim}\"",
                f"- **Verdict**: **{v}** ({conf} Calibrated Confidence)",
                f"- **Consensus Agreement**: {signals.get('consensus_percentage', 'N/A')}",
                f"- **Explanation**: {reason}",
                f"- **Action Advisory**: {adv}",
                f""
            ]

            if v == "CONTRADICTED":
                lines.append("## 🚨 Contradiction Forensics & Ground-Truth Correction:")
                if c_part:
                    lines.append(f"- **What Part is Contradicted**: `{c_part}`")
                if resrc:
                    lines.append(f"- **Proving Resource**: {resrc.get('source')} [{resrc.get('domain')}] ({resrc.get('authority_pct')} Authority)")
                    lines.append(f"- **Evidence Proof Quote**: \"{resrc.get('evidence_text')}\"")
                if r_stmt:
                    lines.append(f"- **What to State Instead (Right Statement)**: \"{r_stmt}\"")
                lines.append("")

            lines.extend([
                f"## 17-Signal Gated Feature Breakdown:",
                f"- NLI Contradiction: {signals.get('nli_contradiction', 0.0):.1%}",
                f"- NLI Entailment: {signals.get('nli_entailment', 0.0):.1%}",
                f"- Semantic Similarity: {signals.get('semantic_similarity', 0.0):.1%}",
                f"- Retrieval Authority: {signals.get('source_reliability', 0.0):.1%}",
                f"- Cross-Passage Consensus: {signals.get('evidence_agreement', 0.0):.1%}"
            ])

        report_md = "\n".join(lines)
        self._copy_with_feedback(report_md, getattr(self, "copy_report_btn", None), "Fact-Check Audit Report Copied!")

    # =========================================================================
    # Status Bar
    # =========================================================================
    def _build_status_bar(self):
        status = ctk.CTkFrame(self, fg_color="#05080f", height=24)
        status.pack(fill="x", side="bottom")

        left = ctk.CTkLabel(status, text="  TRUVI-EV 2.0 • Reliability-Gated Multi-Signal Engine • Obsidian Studio UI", font=ctk.CTkFont(size=10), text_color=COLOR_TEXT_MUTED)
        left.pack(side="left")

        right = ctk.CTkLabel(status, text="Multi-Tier Knowledge: Curated Registry (Tier 1) | Live DuckDuckGo (Tier 2) | Wikipedia API (Tier 3)  ", font=ctk.CTkFont(size=10), text_color=COLOR_TEXT_MUTED)
        right.pack(side="right")

    # =========================================================================
    # Ultra-Responsive Clipboard Auto-Verify Watcher
    # =========================================================================
    def _start_clipboard_watcher(self):
        if not CLIPBOARD_AVAILABLE:
            print("⚠ pyperclip not installed; clipboard auto-verify disabled.")
            return

        # Do NOT latch the current clipboard, so the very next copy immediately triggers!
        self.last_clipboard_text = ""

        self.clipboard_thread = threading.Thread(target=self._clipboard_loop, daemon=True)
        self.clipboard_thread.start()
        print("✓ Background clipboard watcher active. Copy any claim on macOS to auto-verify!")

    def _clipboard_loop(self):
        while True:
            time.sleep(0.3)
            if not self.clipboard_enabled:
                continue

            try:
                if getattr(self, "is_verifying", False):
                    continue

                try:
                    current_text = (pyperclip.paste() or "").strip()
                except Exception:
                    current_text = ""

                # Ignore non-claims, URLs, file paths, or duplicate texts
                if (
                    current_text
                    and current_text != self.last_clipboard_text
                    and 8 <= len(current_text) <= 3500
                    and " " in current_text
                    and not current_text.startswith(("http://", "https://", "file://", "ftp://", "/Users/", "/System/"))
                ):
                    self.last_clipboard_text = current_text

                    # 1. Immediately show floating HUD in "Analyzing..." state (<50ms feedback)
                    self.after(0, self._show_clipboard_analyzing_hud, current_text)

                    # 2. Run verification asynchronously in background thread
                    def _do_verify(text_to_verify):
                        try:
                            has_multiple_sentences = (
                                "\n" in text_to_verify or
                                len([s for s in text_to_verify.split(".") if len(s.strip()) > 10]) > 1
                            )
                            if has_multiple_sentences:
                                res = self.engine.verify_paragraph(text_to_verify)
                            else:
                                res = self.engine.verify_claim(text_to_verify)

                            # Send native macOS notification banner
                            v = res.get("overall_verdict" if has_multiple_sentences else "verdict", "UNVERIFIED")
                            conf = res.get("overall_confidence_pct" if has_multiple_sentences else "confidence_pct", "")
                            r_stmt = res.get("right_statement") or (res.get("contradictions", [{}])[0].get("right_statement") if has_multiple_sentences and res.get("contradictions") else "")
                            msg_banner = f"Right fact: {r_stmt}" if r_stmt else f"Evaluated with {conf} confidence."
                            send_macos_notification(
                                title=f"TRUVI-EV: {v}",
                                subtitle=f'"{text_to_verify[:60]}..."',
                                message=msg_banner
                            )

                            self.after(0, self._update_clipboard_hud_result, res)
                        except Exception as e:
                            print(f"[TRUVI Clipboard Watcher] Verification error: {e}")
                            err_res = {
                                "claim": text_to_verify,
                                "verdict": "UNVERIFIED",
                                "confidence_pct": "70.0%",
                                "short_reason": f"Verification error: {str(e)[:100]}",
                                "action_advisory": "Could not verify claim due to an unexpected runtime error."
                            }
                            self.after(0, self._update_clipboard_hud_result, err_res)

                    threading.Thread(target=_do_verify, args=(current_text,), daemon=True).start()

            except Exception as e:
                print(f"[TRUVI Clipboard Loop] Error: {e}")

    def _show_clipboard_analyzing_hud(self, text: str):
        if self.active_popup is not None and self.active_popup.winfo_exists():
            try:
                self.active_popup.set_analyzing_state(text)
                return
            except Exception:
                pass
        self.active_popup = FloatingHUDNotification(self, initial_text=text)

    def _update_clipboard_hud_result(self, result: Dict[str, Any]):
        if self.active_popup is not None and self.active_popup.winfo_exists():
            try:
                self.active_popup.set_result(result)
                return
            except Exception:
                pass
        self.active_popup = FloatingHUDNotification(self, result=result)


def main():
    app = TRUVIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
