import re
import struct
import time
import numpy as np

from empyric.tools import find_nearest
from empyric.types import Float, Array, Integer, String, Boolean
from empyric.adapters import USB, Socket
from empyric.collection.instrument import Instrument, setter, getter, measurer


class TekTDSScope(Instrument):
    """
    Tektronix oscillscope of the TDS200, TDS1000/2000, TDS1000B/2000B,
    TPS2000 series.

    Although there are up to four channels, the constructor checks the model
    number for the actual number of supported channels. With the 2-channel
    models, any methods relating to channels 3 or 4 will take no action.

    2-Channel models: TXX1001, TXX1002, TXX1012, TXX2002, TXX2012, TXX2022

    4-channel models: TXX2004, TXX2014
    """

    name = "TekScope"

    supported_adapters = ((USB, {"timeout": 10}),)

    knobs = (
        "horz scale",
        "horz position",
        "ch1 scale",
        "ch1 position",
        "ch2 scale",
        "ch2 position",
        "ch3 scale",
        "ch3 position",
        "ch4 scale",
        "ch4 position",
        "trigger level",
    )

    meters = (
        "channel 1",
        "channel 2",
        "channel 3",
        "channel 4",
    )

    channels = 0

    two_channel_models = [1001, 1002, 1012, 2002, 2012, 2022]
    four_channel_models = [2004, 2014]

    def __init__(self, *args, **kwargs):
        # Check for number of channels before standard initialization
        adapter = USB(Instrument(args[0]))

        self.model = int(re.search("\d\d\d\d", adapter.query("*IDN?"))[0])

        adapter.disconnect()

        if self.model in self.two_channel_models:
            self.channels = 2
        elif self.model in self.four_channel_models:
            self.channels = 4

        super().__init__(*args, **kwargs)

    # Horizontal

    @setter
    def set_horz_scale(self, scale: Float):
        self.write("HOR:SCA %.3e" % scale)

    @getter
    def get_horz_scale(self) -> Float:
        return float(self.query("HOR:SCA?"))

    @setter
    def set_horz_position(self, position: Float):
        self.write("HOR:POS %.3e" % position)

    @getter
    def get_horz_position(self) -> Float:
        return float(self.query("HOR:POS?"))

    # Channel 1

    @setter
    def set_ch1_scale(self, scale: Float):
        self.write("CH1:SCA %.3e" % scale)

    @getter
    def get_ch1_scale(self) -> Float:
        return float(self.query("CH1:SCA?"))

    @setter
    def set_ch1_position(self, position: Float):
        self.write("CH1:POS %.3e" % position)

    @getter
    def get_ch1_position(self) -> Float:
        return float(self.query("CH1:POS?"))

    @measurer
    def measure_channel_1(self) -> Array:
        return self._measure_channel(1)

    # Channel 2

    @setter
    def set_ch2_scale(self, scale: Float):
        self.write("CH2:SCA %.3e" % scale)

    @getter
    def get_ch2_scale(self) -> Float:
        return float(self.query("CH2:SCA?"))

    @setter
    def set_ch2_position(self, position: Float):
        self.write("CH2:POS %.3e" % position)

    @getter
    def get_ch2_position(self) -> Float:
        return float(self.query("CH2:POS?"))

    @measurer
    def measure_channel_2(self) -> Array:
        return self._measure_channel(2)

    # Channel 3

    @setter
    def set_ch3_scale(self, scale: Float):
        if self.channels > 2:
            self.write("CH3:SCA %.3e" % scale)
        else:
            return np.nan

    @getter
    def get_ch3_scale(self) -> Float:
        if self.channels > 2:
            return float(self.query("CH3:SCA?"))
        else:
            return np.nan

    @setter
    def set_ch3_position(self, position: Float):
        if self.channels > 2:
            self.write("CH3:POS %.3e" % position)
        else:
            return np.nan

    @getter
    def get_ch3_position(self) -> Float:
        if self.channels > 2:
            return float(self.query("CH3:POS?"))
        else:
            return np.nan

    @measurer
    def measure_channel_3(self) -> Array:
        if self.channels > 2:
            return self._measure_channel(3)
        else:
            return [np.nan]

    # Channel 4

    @setter
    def set_ch4_scale(self, scale: Float):
        if self.channels > 2:
            self.write("CH4:SCA %.3e" % scale)
        else:
            return np.nan

    @getter
    def get_ch4_scale(self) -> Float:
        if self.channels > 2:
            return float(self.query("CH4:SCA?"))
        else:
            return np.nan

    @setter
    def set_ch4_position(self, position: Float):
        if self.channels > 2:
            self.write("CH4:POS %.3e" % position)
        else:
            return np.nan

    @getter
    def get_ch4_position(self) -> Float:
        if self.channels > 2:
            return float(self.query("CH4:POS?"))
        else:
            return np.nan

    @measurer
    def measure_channel_4(self) -> Array:
        if self.channels > 2:
            return self._measure_channel(4)
        else:
            return [np.nan]

    # Trigger

    @setter
    def set_trigger_level(self, level: Float):
        self.write("TRIG:MAI:LEV %.3e" % level)

    @getter
    def get_trigger_level(self) -> Float:
        return float(self.query("TRIG:MAI:LEV?"))

    def _measure_channel(self, n):
        self.write("DAT:ENC ASCI")  # ensure ASCII encoding of data
        self.write("DAT:SOU CH%d" % n)  # switch to channel n

        scale_factor = float(self.query("WFMPRE:YMULT?"))
        zero = float(self.query("WFMPRE:YZERO?"))
        offset = float(self.query("WFMPRE:YOFF?"))

        self.write("ACQ:STATE RUN")  # acquire the waveform

        while int(self.query("BUSY?")):
            time.sleep(0.25)  # wait for acquisition to complete

        normal_timeout = self.adapter.timeout
        self.adapter.timeout = 60

        response = self.query("CURVE?")

        str_data = response.split(",")

        self.adapter.timeout = normal_timeout

        return np.array(
            [(float(datum) - offset) * scale_factor + zero for datum in str_data]
        )


class MulticompProScope(Instrument):
    """
    Multicomp Pro PC Oscilloscope.
    """

    supported_adapters = (
        (Socket, {'timeout': 1.0}),
        (USB, {"delay": 1}),
        # acquisitions can take a long time
    )
    """Supported adapters and options."""

    knobs = (
        "horz scale",
        "horz position",
        "scale ch1",
        "position ch1",
        "scale ch2",
        "position ch2",
        "sweep mode",
        "trigger level",
        "trigger source",
        "sweep mode",
        "resolution",
        "acquire",
        "state",
    )

    meters = (
        "channel 1",
        "channel 2",
    )

    presets = {
        "resolution": 1e4,
        "sweep mode": "SINGLE",
        "trigger source": 1,
    }

    _time_scales = {
        2e-9: "2.0ns",
        5e-9: "5.0ns",
        10e-9: "10ns",
        20e-9: "20ns",
        50e-9: "50ns",
        100e-9: "100ns",
        200e-9: "200ns",
        500e-9: "500ns",
        1e-6: "1.0us",
        2e-6: "2.0us",
        5e-6: "5.0us",
        10e-6: "10us",
        20e-6: "20us",
        50e-6: "50us",
        100e-6: "100us",
        200e-6: "200us",
        500e-6: "500us",
        1e-3: "1.0ms",
        2e-3: "2.0ms",
        5e-3: "5.0ms",
        10e-3: "10ms",
        20e-3: "20ms",
        50e-3: "50ms",
        100e-3: "100ms",
        200e-3: "200ms",
        500e-3: "500ms",
        1: "1.0s",
        2: "2.0s",
        5: "5.0s",
        10: "10s",
        20: "20s",
        50: "50s",
        100: "100s",
    }

    _volt_scales = {
        2e-3: "2mv",
        5e-3: "5mv",
        10e-3: "10mv",
        20e-3: "20mv",
        50e-3: "50mv",
        100e-3: "100mv",
        200e-3: "200mv",
        500e-3: "500mv",
        1.0: "1v",
        2.0: "2v",
        5.0: "5v",
    }

    _volt_div_table = {0: 1e-3, 9: 1.0}

    _resolutions = {1e3: "1K", 1e4: "10K", 1e5: "100K", 1e6: "1M", 1e7: "10M"}

    _sweep_modes = ["AUTO", "NORMAL", "NORM", "SINGLE", "SING"]

    _acquiring = False

    sample_rate = None

    @setter
    def set_horz_scale(self, scale: Float):
        new_time_scale = find_nearest(list(self._time_scales.keys()), scale)

        self.write(":HORI:SCAL " + self._time_scales[new_time_scale])

    @getter
    def get_horz_scale(self) -> Float:
        time_scales_rev = {val: key for key, val in self._time_scales.items()}

        time_scale_str = self.query(":HORI:SCAL?")[:-2]

        return time_scales_rev[time_scale_str]

    @setter
    def set_horz_position(self, offset: Float):
        self.write(":HORI:OFFS " + str(offset))

    @getter
    def get_horz_position(self) -> Float:
        return float(self.query(":HORI:OFFS?")[:-2])

    def _set_scale_ch(self, channel, scale):
        new_volt_scale = find_nearest(list(self._volt_scales.keys()), scale)

        scale_str = self._volt_scales[new_volt_scale]

        self.write((":CH%d:SCAL " % channel) + scale_str)

    def _get_scale_ch(self, channel):
        volt_scale_str = self.query(":CH%d:SCAL?" % channel)[:-2]

        if "mv" in volt_scale_str.lower():
            return 1e-3 * float(volt_scale_str[:-2])
        elif "v" in volt_scale_str.lower():
            return float(volt_scale_str[:-1])
        else:
            return float("nan")

    def _set_position_ch(self, channel, position):
        self.write((":CH%d:OFFS " % channel) + str(position))

    @setter
    def set_scale_ch(self, scale: Float, channel=None):
        self._set_scale_ch(channel, scale)

    @getter
    def get_scale_ch(self, channel=None) -> Float:
        return self._get_scale_ch(channel)

    @setter
    def set_position_ch(self, position: Float, channel=None):
        self._set_position_ch(channel, position)

    @getter
    def get_position_ch(self, channel=None) -> Float:
        return float(self.query(":CH%d:OFFS?" % channel)[:-2])

    @setter
    def set_sweep_mode(self, mode: String):
        if mode.upper() not in self._sweep_modes:
            raise ValueError(
                f"Invalid sweep mode! "
                f"Sweep mode must be one of: {', '.join(self._sweep_modes)}"
            )

        self.write(":TRIG:SING:SWE " + mode.upper())

    @getter
    def get_sweep_mode(self) -> String:
        return self.query(":TRIG:SING:SWE?")[:-2].upper()

    @setter
    def set_trigger_source(self, source: String):
        self.write(":TRIG:SING:EDGE:SOUR CH%d" % int(source))

    @getter
    def get_trigger_source(self) -> String:
        source = self.query(":TRIG:SING:EDGE:SOUR?")[:-2]

        if source == "CH1":
            return 1
        elif source == "CH2":
            return 2

    @setter
    def set_trigger_level(self, level_volts: Float):
        trigger_source = self.get_trigger_source()

        scale = self._get_scale_ch(trigger_source)

        level_div = str(level_volts / scale)

        self.write(":TRIG:SING:EDGE:LEV " + level_div)

    @getter
    def get_trigger_level(self) -> Float:
        trigger_source = self.get_trigger_source()

        scale = self._get_scale_ch(trigger_source)

        return float(self.query(":TRIG:SING:EDGE:LEV?")[:-2]) * scale

    @measurer
    def measure_trigger_status(self) -> String:
        return self.query(":TRIG:STATUS?")[:-2]

    @setter
    def set_resolution(self, resolution: Integer):
        if resolution not in self._resolutions:
            raise ValueError(
                f"Invalid memory depth! "
                f"Memory depth must be one of: {', '.join(str(self._resolutions))}"
            )

        self.write(":ACQ:DEPMEM " + self._resolutions[resolution])

    @getter
    def get_resolution(self) -> Integer:
        res_rev = {val: key for key, val in self._resolutions.items()}

        return res_rev[self.query(":ACQ:DEPMEM?")[:-2]]

    @setter
    def set_state(self, state: String):
        if state.upper() in ["RUN", "STOP"]:
            self.write(f":{state.upper()}")

    @setter
    def set_acquire(self, _: Boolean):
        self._acquiring = True

        self.set_sweep_mode("SINGLE")

        trig_status = self.measure_trigger_status()

        if trig_status == "TRIG":
            # measurement has been triggered
            pass
        else:
            # sweep needs to be reset
            time.sleep(0.1)
            self.set_acquire(1)

    def _read_preamble(self, channel):
        info_dict = {}

        preamble = self.query(":WAV:PRE?", decode=False)

        header, info = preamble[:11], preamble[11:]

        if header != b"#9000000000":  # indicates empty data buffer
            try:
                # First 8 bytes is an integer signature, used for verification
                ver_int = 651058244139746640
                verified = struct.unpack("<q", info[:8])[0] == ver_int

                if verified:
                    if channel == 1:
                        scale = self._volt_div_table[
                            struct.unpack("<h", info[260:262])[0]
                        ]
                        zero = struct.unpack("<f", info[268:272])[0]
                    else:
                        scale = self._volt_div_table[
                            struct.unpack("<h", info[262:264])[0]
                        ]
                        zero = struct.unpack("<f", info[272:276])[0]

                    sample_rate = struct.unpack("<f", info[316:320])[0] * 1e6
                    # Hz

                    info_dict["scale"] = scale
                    info_dict["zero"] = zero
                    info_dict["sample rate"] = sample_rate
                else:
                    print(
                        f"Warning: {self.name} received unverified "
                        f"signature when reading from channel {channel}"
                    )

            except struct.error:
                print(
                    f"Warning: {self.name} received truncated preamble when "
                    f"reading from channel {channel}"
                )

        self.sample_rate = info_dict["sample rate"]

        return info_dict

    def _read_data(self, channel):
        info = self._read_preamble(channel)

        if info:
            scale = info["scale"]
            zero = info["zero"]
        else:
            return None

        # Only 256k data points can be read at a time
        resolution = self.get_resolution()

        if resolution < 256000:  # can get all data in one pass
            self.write(":WAV:RANG 0, %d" % resolution)

            raw_response = self.query(":WAV:FETC?", decode=False)

            header, byte_data = raw_response[:11], raw_response[11:]

            data_len = int(header[2:]) // 2

            voltages = scale * (
                np.array(struct.unpack(f"<{data_len}h", byte_data), dtype=float) / 6400
                - zero
            )

        else:  # need to read data in chunks; max chunk size is 256k
            offset, size = 0, 200000

            voltages = np.empty(resolution)

            while offset + size <= resolution:
                self.write(":WAV:RANG %d, %d" % (offset, size))

                raw_response = self.query(":WAV:FETC?", decode=False)

                header, byte_data = raw_response[:11], raw_response[11:]

                data_len = int(header[2:]) // 2

                voltages[offset: offset + size] = scale * (
                    np.array(struct.unpack(f"<{data_len}h", byte_data), dtype=float)
                    / 6400
                    - zero
                )

                offset += size

        if len(voltages) != resolution:
            print(
                f"Warning: {self.name} received truncated data "
                f"for channel {channel}; discarding"
            )
            return None

        return voltages

    @measurer
    def measure_channel_1(self) -> Array:
        if not self._acquiring:
            self.set_acquire(True)

        self._acquiring = False

        self.write(":WAV:BEG CH1")

        data = self._read_data(1)

        self.write(":WAV:END CH1")

        return data

    @measurer
    def measure_channel_2(self) -> Array:
        if not self._acquiring:
            self.set_acquire(True)

        self._acquiring = False

        self.write(":WAV:BEG CH2")

        data = self._read_data(2)

        self.write(":WAV:END CH2")

        return data


class SiglentSDS1000(Instrument):
    """
    Siglent SDS1000 series digital oscilloscope
    """

    name = "SiglentSDS1000"

    supported_adapters = (
        (
            Socket,
            {
                "write_termination": "\n",
                "read_termination": "\n",
                "timeout": 60,  # waveform retrieval may take some extra time
            },
        ),
    )

    knobs = (
        "horz scale",
        "horz position",
        "ch1 scale",
        "ch1 position",
        "ch2 scale",
        "ch2 position",
        "ch3 scale",
        "ch3 position",
        "ch4 scale",
        "ch4 position",
        "trigger source",
        "trigger level",
        "trigger mode",
        "arm",
        "acquire mode",
        "memory size",
        "averages",
    )

    meters = ("ch1 waveform", "ch2 waveform", "ch3 waveform", "ch4 waveform")

    horz_scales = [
        1e-9,
        2e-9,
        5e-9,
        10e-9,
        20e-9,
        50e-9,
        100e-9,
        200e-9,
        500e-9,
        1e-6,
        2e-6,
        5e-6,
        10e-6,
        20e-6,
        50e-6,
        100e-6,
        200e-6,
        500e-6,
        1e-3,
        2e-3,
        5e-3,
        10e-3,
        20e-3,
        50e-3,
        100e-3,
        200e-3,
        500e-3,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,  # seconds
    ]

    _trg_src = None

    # Time axis control
    @setter
    def set_horz_scale(self, scale: Float):
        nearest_scale = find_nearest(self.horz_scales, scale, overestimate=True)

        self.write(f"TDIV {nearest_scale}S")

        return nearest_scale

    @getter
    def get_horz_scale(self) -> Float:
        response = self.query("TDIV?").split("TDIV ")[-1][:-1]

        try:
            response = float(response)
        except ValueError:
            return np.nan

        return response

    @setter
    def set_horz_position(self, position: Float):
        self.write(f"TRDL {position}S")

    @getter
    def get_horz_position(self) -> Float:
        response = self.query("TRDL?").split("TRDL ")[-1][:-1]

        try:
            response = float(response)
        except ValueError:
            return np.nan

        return response

    # Channel control base functions
    def _set_chn_scale(self, n, scale):
        self.write("C%d:VDIV %.3eV" % (n, float(scale)))

    def _get_chn_scale(self, n):
        response = self.query("C%d:VDIV?" % n).split("C%d:VDIV " % n)[-1][:-1]

        try:
            response = float(response)
        except ValueError:
            return np.nan

        return response

    def _set_chn_position(self, n, position):
        self.write("C%d:OFST %.3eV" % (n, float(position)))

    def _get_chn_position(self, n):
        response = self.query("C%d:OFST?" % n).split("C%d:OFST " % n)[-1][:-1]

        try:
            response = float(response)
        except ValueError:
            return np.nan

        return response

    # Channel 1 control
    @setter
    def set_ch1_scale(self, scale: Float):
        self._set_chn_scale(1, scale)

    @getter
    def get_ch1_scale(self) -> Float:
        return self._get_chn_scale(1)

    @setter
    def set_ch1_position(self, position: Float):
        self._set_chn_position(1, position)

    @getter
    def get_ch1_position(self) -> Float:
        return self._get_chn_position(1)

    # Channel 2 control
    @setter
    def set_ch2_scale(self, scale: Float):
        self._set_chn_scale(2, scale)

    @getter
    def get_ch2_scale(self) -> Float:
        return self._get_chn_scale(2)

    @setter
    def set_ch2_position(self, position: Float):
        self._set_chn_position(2, position)

    @getter
    def get_ch2_position(self) -> Float:
        return self._get_chn_position(2)

    # Channel 3 control
    @setter
    def set_ch3_scale(self, scale: Float):
        self._set_chn_scale(3, scale)

    @getter
    def get_ch3_scale(self) -> Float:
        return self._get_chn_scale(3)

    @setter
    def set_ch3_position(self, position: Float):
        self._set_chn_position(3, position)

    @getter
    def get_ch3_position(self) -> Float:
        return self._get_chn_position(3)

    # Channel 4 control
    @setter
    def set_ch4_scale(self, scale: Float):
        self._set_chn_scale(4, scale)

    @getter
    def get_ch4_scale(self) -> Float:
        return self._get_chn_scale(4)

    @setter
    def set_ch4_position(self, position: Float):
        self._set_chn_position(4, position)

    @getter
    def get_ch4_position(self) -> Float:
        return self._get_chn_position(4)

    # Trigger control
    @setter
    def set_trigger_level(self, level: Float):
        trg_src = self.get_trigger_source()

        self.write(f"C{trg_src}:TRLV {level}V")

    @getter
    def get_trigger_level(self) -> Float:
        trg_src = self.get_trigger_source()

        if trg_src > 0:
            response = self.query(f"C{trg_src}:TRLV?").split("TRLV ")[-1][:-1]
        else:
            return np.nan

        try:
            response = float(response)
        except ValueError:
            return np.nan

        return response

    @setter
    def set_trigger_source(self, source: Integer):
        self.write(f"TRSE EDGE,SR,C{int(source)},OFF")

    @getter
    def get_trigger_source(self) -> Integer:
        def validator(response):
            return "SR,C" in response

        response = self.query("TRSE?", validator=validator).split("SR,C")[-1][0]

        try:
            response = int(response)
        except ValueError:
            return -1

        return response

    @setter
    def set_trigger_mode(self, mode: String):
        if mode.upper() in ["AUTO", "NORM", "SINGLE", "STOP"]:
            self.write(f"TRMD {mode.upper()}")
        else:
            return self.get_trigger_mode()

    @getter
    def get_trigger_mode(self) -> String:
        try:
            return self.query("TRMD?")[5:]
        except IndexError:
            return None

    @setter
    def set_acquire_mode(self, mode: String):
        if mode.upper() in ["SAMPLING", "PEAK_DETECT", "HIGH_RES"]:
            self.write(f"ACQW {mode.upper()}")
        elif mode.upper() == "AVERAGE":
            if not hasattr(self, "averages"):
                self.set_averages(4)

            self.write(f"ACQW AVERAGE,{int(self.averages)}")

        else:
            return None

    @getter
    def get_acquire_mode(self) -> String:
        response = self.query("ACQW?").split("ACQW ")[-1]

        if "AVERAGE" in response:  # also includes # of averages
            return "AVERAGE"
        else:
            return response

    @setter
    def set_memory_size(self, size: Integer):
        self.write(f"MSIZ {int(size)}")

        return self.get_memory_size()

    @getter
    def get_memory_size(self) -> Integer:
        response = self.query("MSIZ?").split("MSIZ ")[-1]

        if "K" in response:
            response = float(response.replace("K", "e3"))
        elif "M" in response:
            response = float(response.replace("M", "e6"))

        try:
            response = int(response)
        except ValueError:
            return -1

        return response

    @setter
    def set_averages(self, averages: Integer):
        valid = [4] + [2**i for i in range(4, 11)]

        averages = find_nearest(valid, averages)

        self.write(f"AVGA {averages}")

    @getter
    def get_averages(self) -> Integer:
        response = self.query("AVGA?").split("AVGA ")[-1]

        try:
            response = int(response)
        except ValueError:
            return -1

        return response

    # Channel measurements
    def _measure_chn_waveform(self, n):
        ready = False
        while not ready:
            time.sleep(0.25)
            ready = "1" in self.query("*OPC?")

        # Enable channel if needed
        ch_enabled_query = self.query("C%d:TRA?" % n)
        if "ON" not in ch_enabled_query:
            print(f"Warning: CH{n} is not enabled; cannot retrieve waveform")
            return None

        self.write("WFSU SP,0,NP,0,FP,0")  # setup to get all data points

        def is_terminated(message):
            try:
                size = int(message[13:22])
            except ValueError:
                return False

            data_length = len(message[22:-2])

            if data_length == size:
                return True
            else:
                return False

        self.adapter.read_termination = is_terminated
        max_attempts = self.adapter.max_attempts
        self.adapter.max_attempts = 1

        def validator(response):
            """
            Checks for correct header, size, message length and termination.
            """

            if response == b"":
                return True  # no data

            if response is None:
                return False
            elif len(response) == 0:
                return False

            header = response[:13]
            if header != (b"C%d:WF DAT2,#9" % n):
                print("invalid header", header)
                return False

            size = response[13:22]
            try:
                size = int(size)
            except ValueError:
                print("invalid data size", size)
                return False

            data = response[22:-2]
            if len(data) != size:
                print("truncated data", len(data))
                return False

            termination = response[-2:]
            if termination != b"\n\n":
                print("no termination", response[-2:])
                return False

            return True

        response = self.query("C%d:WF? DAT2" % n, decode=False, validator=validator)

        if response == b"":
            return None

        self.adapter.read_termination = "\n"  # restore normal read termination
        self.adapter.max_attempts = max_attempts

        header, size, waveform = response[:13], response[13:22], response[22:-2]
        size = int(size)

        # convert bytes to integers
        waveform = np.array(struct.unpack("b" * size, waveform), dtype=np.float64)

        scale = self._get_chn_scale(n)
        pos = self._get_chn_position(n)

        waveform = (scale / 25.0) * waveform - pos

        return waveform

    @measurer
    def measure_ch1_waveform(self) -> Array:
        return self._measure_chn_waveform(1)

    @measurer
    def measure_ch2_waveform(self) -> Array:
        return self._measure_chn_waveform(2)

    @measurer
    def measure_ch3_waveform(self) -> Array:
        return self._measure_chn_waveform(3)

    @measurer
    def measure_ch4_waveform(self) -> Array:
        return self._measure_chn_waveform(4)

    @measurer
    def measure_sample_rate(self) -> Float:
        response = self.query("SARA?")[5:-4]

        try:
            response = float(response)
        except ValueError:
            return np.nan

        return response
