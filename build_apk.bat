@echo off
set "ANDROID_HOME=C:\Users\c_r_a\AppData\Local\Android\Sdk"
set "JAVA_HOME=C:\Program Files\Android\Android Studio\jbr"
set "PLATFORM=android-36.1"
set "BUILD_TOOLS=36.0.0"
set "PATH=%JAVA_HOME%\bin;%ANDROID_HOME%\build-tools\%BUILD_TOOLS%;%ANDROID_HOME%\platform-tools;%PATH%"

echo Building PT Capture APK for %PLATFORM%...
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Clean old classes
if exist com rd /s /q com

REM Compile Java
echo Compiling Java source...
javac -d . -cp ".;%ANDROID_HOME%\platforms\%PLATFORM%\android.jar" ..\CaptureActivity.java
if %errorlevel% neq 0 (
    echo ERROR: Java compilation failed
    cd ..
    exit /b 1
)

REM Create R.java (minimal)
echo Creating R.java...
echo package com.pt.capture; public final class R { public static final class string { public static final int app_name = 0x7f040000; } } > R.java
javac -d . -cp ".;%ANDROID_HOME%\platforms\%PLATFORM%\android.jar" R.java

REM Generate classes.dex using d8
echo Generating classes.dex...
java -cp "%ANDROID_HOME%\build-tools\%BUILD_TOOLS%\lib\d8.jar;." com.android.tools.r8.D8 --lib "%ANDROID_HOME%\platforms\%PLATFORM%\android.jar" --output . com\pt\capture\*.class

if not exist classes.dex (
    echo ERROR: classes.dex generation failed
    cd ..
    exit /b 1
)

REM Create APK structure
mkdir res
mkdir res\values
echo ^<?xml version="1.0" encoding="utf-8"?^>^<resources^>^<string name="app_name"^>PT Capture^</string^>^</resources^> > res\values\strings.xml

REM Package resources
aapt package -f -M ..\AndroidManifest.xml -S res -I "%ANDROID_HOME%\platforms\%PLATFORM%\android.jar" -F PTCapture.unsigned.apk

REM Add classes.dex
aapt add PTCapture.unsigned.apk classes.dex

REM Sign APK (debug key)
echo Signing APK...
REM jarsigner -verbose -sigalg SHA256withRSA -digestalg SHA256 -keystore "%USERPROFILE%\.android\debug.keystore" -storepass android PTCapture.unsigned.apk androiddebugkey
call apksigner sign --ks "%USERPROFILE%\.android\debug.keystore" --ks-pass pass:android --out ..\ptcapture.apk PTCapture.unsigned.apk


cd ..
echo.
echo APK built successfully: ptcapture.apk
