# Network Camera Address Recovery

Plant Observatory can follow a DHCP-renumbered RTSP camera by its MAC address.
The feature scans only bounded private IPv4 networks assigned to the PC, checks
known camera and RTSP ports, and confirms an RTSP response before accepting a
candidate. A new address is saved only when exactly one responding RTSP endpoint
has the paired MAC.

## Save Camera Identity

1. Open **Settings > Network Camera Recovery**.
2. Enter the camera's MAC address once and choose **Save MAC & Find**. The MAC can
   usually be found in the hotspot's connected-device list or router client list.
3. Plant Observatory stores the pairing and discovered address in
   `configs/network_camera_discovery.json` under its active data root. It does
   not rewrite `network_cameras.local.yaml` or copy camera credentials.

After pairing, a failed reachability probe or a MAC mismatch at the saved address
starts an asynchronous scan, limited to one scan per camera per 90 seconds.
**Find Camera** starts an immediate scan.
If no unique exact-MAC RTSP match is found, the configured address is left alone.

## Limits

- Only local private IPv4 networks with at most 512 addresses each are scanned;
  total scan size is capped at 1024 host addresses.
- The current Windows implementation reads adapters and MAC neighbors through
  PowerShell networking cmdlets. Linux uses `ip` JSON output and its neighbor
  table.
- RTSP port candidates are 554 and 8554. Port 80, 5000, 8080, and 8899 can help
  identify responsive hosts but are not sufficient to update an RTSP camera.
- Discovery does not infer credentials or stream paths. Those remain in the
  local camera configuration. An RTSP OPTIONS response confirms the service,
  not that authentication and frame decoding succeed.
- A DHCP reservation in the router/hotspot is still the simplest way to keep an
  address stable when that feature is available.

Discovery endpoints accept only configured camera IDs and never accept a
browser-supplied host or subnet. Pairing and manual scans require a same-origin
request. API results omit credentials and full stream URLs.