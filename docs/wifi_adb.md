# Optional Wi-Fi ADB

Architextures can use Android phones over USB ADB or Wi-Fi ADB. USB remains the
setup and recovery path. Wi-Fi is an optional transport for phones beyond cable
reach, including phones connected to a Windows PC hotspot while the PC uses
Ethernet.

## Recommended setup

1. Enable Developer options and USB debugging on the phone.
2. Connect the phone to the PC hotspot.
3. Connect it by USB once and accept the debugging authorization.
4. In **Settings > Wi-Fi ADB**, choose the USB serial and press **Prepare USB
   Phone for Wi-Fi**.
5. Save the detected `IP:5555` endpoint.

The endpoint is stored in `configs/android_devices.local.yaml`. Architextures
periodically reconnects enabled entries and then uses the normal ADB device ID,
such as `192.168.137.42:5555`, everywhere capture already supports USB serials.

For Android 11 and newer, the same panel also accepts the pairing address, port,
and code shown by Android's **Pair device with pairing code** screen. Pair first,
then connect and save the separate wireless-debugging connection address shown
by Android. Manual IP entry avoids depending on mDNS discovery.

## Reliability notes

- Keep USB available for authorization, reboot recovery, and APK servicing.
- Reserve phone addresses in the hotspot when possible.
- Allow `adb.exe` on Windows private networks.
- Some older phones stop listening on TCP after reboot or aggressive power
  saving. Reconnect USB and run legacy preparation again when that happens.
- The capture APK still takes and stores each image locally. ADB is used for
  control, synchronization, updates, and diagnostics.
