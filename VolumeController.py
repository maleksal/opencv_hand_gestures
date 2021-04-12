"""
Volume Controller Module
"""

from ctypes import cast, POINTER

import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class Controller:
    def __init__(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_,
            CLSCTX_ALL,
            None)

        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        vrange = self.volume.GetVolumeRange()
        self.min_volume = vrange[0]
        self.max_volume = vrange[1]

    def set_volume(self, length, volume_range):
        """ set volume
        """
        vol = np.interp(length, volume_range, [self.min_volume,
                                            self.max_volume])
        self.volume.SetMasterVolumeLevel(vol, None)
