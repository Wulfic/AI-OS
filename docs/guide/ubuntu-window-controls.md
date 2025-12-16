# Ubuntu/GNOME Window Controls Guide

## Issue: Missing Minimize/Maximize Buttons

On Ubuntu and other distributions using GNOME desktop environment, you may notice that application windows only show a **Close** button by default, without **Minimize** and **Maximize** buttons.

This is **not a bug** in AI-OS - it's a design choice made by GNOME developers. GNOME expects users to use the Activities overview (Super key) or Alt+Tab to switch between applications instead of minimizing them.

## Solution: Enable Minimize/Maximize Buttons

There are two ways to enable these buttons:

### Method 1: Using GNOME Tweaks (GUI)

1. Install GNOME Tweaks if not already installed:
   ```bash
   sudo apt install gnome-tweaks
   ```

2. Open GNOME Tweaks from your applications menu

3. Navigate to **Window Titlebars** from the left sidebar

4. Under **Titlebar Buttons**, toggle on:
   - **Minimize**
   - **Maximize**

5. The buttons will appear immediately on all application windows

**Optional**: You can also change the button placement from right to left if you prefer macOS-style window controls.

### Method 2: Using Command Line

Run this command in a terminal:

```bash
gsettings set org.gnome.desktop.wm.preferences button-layout ":minimize,maximize,close"
```

**To move buttons to the left side** (like macOS):
```bash
gsettings set org.gnome.desktop.wm.preferences button-layout "close,minimize,maximize:"
```

**To disable the buttons again** (restore GNOME default):
```bash
gsettings set org.gnome.desktop.wm.preferences button-layout ":close"
```

## Technical Details

AI-OS is properly configured with:
- Window decorations enabled (`overrideredirect=False`)
- Resizable window enabled
- Standard Tkinter window configuration

The appearance of minimize/maximize buttons is controlled entirely by your desktop environment (GNOME, KDE, XFCE, etc.), not by the application itself.

## Other Desktop Environments

- **KDE Plasma**: Minimize/maximize buttons are shown by default
- **XFCE**: Minimize/maximize buttons are shown by default  
- **Cinnamon**: Minimize/maximize buttons are shown by default
- **MATE**: Minimize/maximize buttons are shown by default

Only GNOME disables these buttons by default.

## Additional Resources

- [GNOME Design Philosophy](https://wiki.gnome.org/Design)
- [It's FOSS Guide: Add Minimize and Maximize Buttons in GNOME](https://itsfoss.com/gnome-minimize-button/)
- [Ask Ubuntu: How to bring back minimize and maximize buttons](https://askubuntu.com/questions/651347/how-to-bring-back-minimize-and-maximize-buttons-in-gnome-3)
