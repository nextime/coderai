#!/bin/sh
# Lock every AMD GPU to its top power/clock state so the amdgpu driver never
# switches DPM power states mid-kernel — those transitions hang Polaris/GCN
# cards under sustained Vulkan compute. Matched by PCI vendor id (0x1002), so
# it is robust to DRM card renumbering across GPU resets/reboots.
#
# Install (root):
#   sudo cp amdgpu-dpm-high.sh /usr/local/sbin/ && sudo chmod 755 /usr/local/sbin/amdgpu-dpm-high.sh
# Persist at boot — add before `exit 0` in /etc/rc.local:
#   /usr/local/sbin/amdgpu-dpm-high.sh || true
for c in /sys/class/drm/card*; do
    [ "$(cat "$c/device/vendor" 2>/dev/null)" = "0x1002" ] && \
        echo high > "$c/device/power_dpm_force_performance_level" 2>/dev/null
done
exit 0
