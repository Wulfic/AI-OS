using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

namespace AIOSLauncher
{
    class Program
    {
        // Windows API for console management
        [DllImport("kernel32.dll")]
        private static extern bool FreeConsole();
        
        [DllImport("kernel32.dll")]
        private static extern bool AttachConsole(int dwProcessId);
        
        private const int ATTACH_PARENT_PROCESS = -1;

        static void Main(string[] args)
        {
            try
            {
                string currentDir = AppDomain.CurrentDomain.BaseDirectory;
                string batPath = Path.Combine(currentDir, "launcher.bat");
                
                // Check for GUI mode (default when launched from shortcut or double-click)
                bool isGuiMode = false;
                if (args.Length > 0 && args[0].Equals("gui", StringComparison.OrdinalIgnoreCase))
                {
                    isGuiMode = true;
                }

                // If not found in current dir, check if we are in a bin folder or similar, 
                // but the plan is to put the EXE in the root next to launcher.bat.
                
                if (!File.Exists(batPath))
                {
                    // Try to attach to parent console to show error
                    AttachConsole(ATTACH_PARENT_PROCESS);
                    Console.WriteLine("Error: launcher.bat not found.");
                    Console.WriteLine("Expected at: " + batPath);
                    Console.WriteLine("Press any key to exit...");
                    Console.ReadKey();
                    return;
                }

                ProcessStartInfo startInfo = new ProcessStartInfo();
                startInfo.FileName = batPath;
                
                // Reconstruct arguments
                // We need to be careful with quoting. 
                // A simple join might break paths with spaces if not quoted.
                // But args array usually has quotes stripped.
                // We'll wrap each arg in quotes.
                string arguments = "";
                if (args.Length > 0)
                {
                    for (int i = 0; i < args.Length; i++)
                    {
                        arguments += "\"" + args[i] + "\" ";
                    }
                }
                startInfo.Arguments = arguments.Trim();
                
                startInfo.WorkingDirectory = currentDir;
                
                if (isGuiMode)
                {
                    // For GUI mode: Detach from any console to prevent console window flash
                    // and avoid stdout/stderr issues that can crash Tkinter
                    FreeConsole();
                    
                    // Use CreateNoWindow to run batch silently, redirect to avoid issues
                    startInfo.UseShellExecute = false;
                    startInfo.CreateNoWindow = true;
                    startInfo.RedirectStandardOutput = true;
                    startInfo.RedirectStandardError = true;
                    startInfo.RedirectStandardInput = true;
                    
                    Process p = Process.Start(startInfo);
                    
                    // Don't wait - let GUI run independently and exit wrapper immediately
                    // This prevents the console from lingering
                    // Note: GUI crashes would now need to be diagnosed through crash logs
                }
                else
                {
                    // For CLI mode: keep console visible and wait
                    startInfo.UseShellExecute = true;
                    
                    Process p = Process.Start(startInfo);
                    p.WaitForExit();
                }
            }
            catch (Exception ex)
            {
                // Try to attach to parent console to show error
                AttachConsole(ATTACH_PARENT_PROCESS);
                Console.WriteLine("Error launching AI-OS: " + ex.Message);
                Console.WriteLine("Press any key to exit...");
                try { Console.ReadKey(); } catch { }
            }
        }
    }
}
