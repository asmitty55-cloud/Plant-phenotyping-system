Set shell = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")
root = fs.GetParentFolderName(WScript.ScriptFullName)
pythonw = root & "\.venv\Scripts\pythonw.exe"

If Not fs.FileExists(pythonw) Then
    pythonw = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Python\Python312\pythonw.exe"
End If

If Not fs.FileExists(pythonw) Then
    pythonw = "pythonw.exe"
End If

shell.Run """" & pythonw & """ """ & root & "\launch_observatory.py""", 0, False
