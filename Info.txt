go into phone settings and find the build number. press it 7 times to endable developer mode

enable USB debugging and keep screen on when plugged in(may not be neccesary?)
connect phone to pc and run program on pc.
on phone, check always grant permission on the prompt.

to remove/reinstal APK on previously used phones:

Use powershell:

adb -s DEVICE_ID install -r -t -d C:\dev\pt\ptcapture.apk
If that fails on an older Android version because of flags, use:

powershell

adb -s DEVICE_ID install -r -t C:\dev\pt\ptcapture.apk
Or fully clean install:

powershell

adb -s DEVICE_ID uninstall com.pt.capture
adb -s DEVICE_ID install -r -t C:\dev\pt\ptcapture.apk