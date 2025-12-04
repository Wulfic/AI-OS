using System;
using System.Diagnostics;
using System.IO;

namespace AIOSLauncher
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                string currentDir = AppDomain.CurrentDomain.BaseDirectory;
                string batPath = Path.Combine(currentDir, "launcher.bat");

                // If not found in current dir, check if we are in a bin folder or similar, 
                // but the plan is to put the EXE in the root next to launcher.bat.
                
                if (!File.Exists(batPath))
                {
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
                startInfo.UseShellExecute = true; 

                Process p = Process.Start(startInfo);
                // We don't necessarily need to wait for exit if we want to detach, 
                // but keeping the wrapper alive until the batch finishes is fine.
                p.WaitForExit();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error launching AI-OS: " + ex.Message);
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
            }
        }
    }
}
