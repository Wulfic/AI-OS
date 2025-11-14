"""
HuggingFace Authentication

Functions for managing HuggingFace Hub authentication.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional, Tuple

# Import safe variable wrappers
from ...utils import safe_variables

# Lazy imports from hf_search
from .hf_search import login, whoami, HfFolder


def get_hf_login_status() -> Tuple[bool, str]:
    """
    Check HuggingFace login status.
    
    Returns:
        Tuple of (is_logged_in, status_message)
    """
    if HfFolder is None or whoami is None:
        return (False, "HF library not available")
    
    try:
        token = HfFolder.get_token()
        if token:
            try:
                user_info = whoami(token=token)
                # Handle different response formats: dict with 'name', 'id', or other
                if isinstance(user_info, dict):
                    username = user_info.get('name') or user_info.get('id') or user_info.get('username', 'Unknown')
                else:
                    username = str(user_info) if user_info else 'Unknown'
                return (True, f"✅ {username}")
            except Exception:
                return (False, "❌ Token error")
        else:
            return (False, "Not logged in")
    except Exception:
        return (False, "Error")


def logout_from_hf(log_func: Callable[[str], None]) -> bool:
    """
    Logout from HuggingFace.
    
    Args:
        log_func: Function to call for logging messages
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if HfFolder is not None:
            HfFolder.delete_token()
        log_func("✓ Successfully logged out of HuggingFace")
        return True
    except Exception as e:
        messagebox.showerror("Logout Failed", f"Failed to logout: {e}")
        return False


def show_login_dialog(parent: tk.Widget, log_func: Callable[[str], None]) -> Optional[str]:
    """
    Show dialog to login to HuggingFace.
    
    Args:
        parent: Parent widget
        log_func: Function to call for logging messages
        
    Returns:
        Username if successful, None otherwise
    """
    if login is None or whoami is None:
        messagebox.showerror(
            "Library Missing",
            "huggingface_hub library is not installed.\n\n"
            "Install it with: pip install huggingface_hub"
        )
        return None
    
    # Create login dialog
    dialog = tk.Toplevel(parent)
    dialog.title("🔐 Login to HuggingFace")
    dialog.geometry("500x350")
    dialog.transient(parent)
    dialog.grab_set()
    
    result_username = None
    
    # Instructions
    instructions = ttk.Label(
        dialog,
        text="To access gated datasets (like The Stack), you need to login:\n\n"
             "1. Go to: https://huggingface.co/settings/tokens\n"
             "2. Create a new token (or copy existing one)\n"
             "3. Paste it below\n"
             "4. Your token will be securely stored",
        justify="left",
        wraplength=450
    )
    instructions.pack(pady=15, padx=15)
    
    # Token entry frame
    token_frame = ttk.Frame(dialog)
    token_frame.pack(pady=10, padx=15, fill="x")
    
    ttk.Label(token_frame, text="Access Token:").pack(anchor="w")
    token_entry = ttk.Entry(token_frame, show="*", width=50)
    token_entry.pack(fill="x", pady=5)
    token_entry.focus()
    
    # Show token checkbox
    show_var = safe_variables.BooleanVar(value=False)
    def toggle_show():
        token_entry.config(show="" if show_var.get() else "*")
    
    ttk.Checkbutton(
        token_frame,
        text="Show token",
        variable=show_var,
        command=toggle_show
    ).pack(anchor="w")
    
    # Status label
    status_label = ttk.Label(dialog, text="", font=("", 9, "italic"))
    status_label.pack(pady=10)
    
    # Button frame
    btn_frame = ttk.Frame(dialog)
    btn_frame.pack(pady=15)
    
    def do_login():
        nonlocal result_username
        token = token_entry.get().strip()
        if not token:
            status_label.config(text="❌ Please enter a token", foreground="red")
            return
        
        # Use gray for "in progress" status to be theme-friendly
        status_label.config(text="⏳ Validating token...", foreground="gray")
        dialog.update()
        
        try:
            # Login - this saves the token
            login(token=token, add_to_git_credential=True)  # type: ignore
            
            # Verify the token works by calling whoami
            user_info = whoami(token=token)  # type: ignore
            
            # Handle different response formats
            if isinstance(user_info, dict):
                username = user_info.get('name') or user_info.get('id') or user_info.get('username', 'Unknown')
            else:
                username = str(user_info) if user_info else 'Unknown'
            
            status_label.config(text=f"✅ Success! Logged in as: {username}", foreground="green")
            log_func(f"✅ HuggingFace: Logged in as {username}")
            result_username = username
            
            # Close dialog after 1 second
            dialog.after(1000, dialog.destroy)
        except Exception as e:
            # Provide detailed error message
            error_type = type(e).__name__
            error_msg = str(e)[:100]
            status_label.config(text=f"❌ {error_type}: {error_msg}", foreground="red")
            log_func(f"❌ HuggingFace login failed ({error_type}): {e}")
    
    def do_cancel():
        dialog.destroy()
    
    ttk.Button(btn_frame, text="🔐 Login", command=do_login).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="❌ Cancel", command=do_cancel).pack(side="left", padx=5)
    
    # Open browser button
    def open_hf_tokens():
        import webbrowser
        webbrowser.open("https://huggingface.co/settings/tokens")
    
    ttk.Button(
        dialog,
        text="🌐 Open HuggingFace Tokens Page",
        command=open_hf_tokens
    ).pack(pady=5)
    
    # Bind Enter key
    token_entry.bind("<Return>", lambda e: do_login())
    
    # Wait for dialog to close
    dialog.wait_window()
    
    return result_username
