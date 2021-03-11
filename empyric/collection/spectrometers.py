import time
import struct
from empyric.adapters import *
from empyric.collection.instrument import *

class SRSRGA(Instrument):
    """
    SRS Residual Gas Analyzer, a quadrupole mass spectrometer with mass ranges from 100 to 300 amu
    """

    name = 'SRS-RGA'

    supported_adapters = (
        (Serial, {'baud_rate': 28800, 'timeout': None, 'termination': b'\n\r'}),
    )

    knobs = (
        'filament current',
        'mass',
        'masses',
        'ppsf',  # partial pressure sensitivity factor
        'tpsf'  # total pressure sensitivity factor
    )

    presets = {
        'filament current': 1
    }

    postsets = {
        'filament current': 0
    }

    meters = {
        'filament current',
        'single',
        'spectrum',
        'total pressure'
    }

    ppsf = 0.119
    tpsf = 0.0023

    @setter
    def set_filament_current(self, current):
        # current is in mA

        if current >= 3.5:
            self.query('FL3.5\r')
        elif current <= 0:
            self.query('FL0\r')
        else:
            self.query('FL'+f'{np.round(float(current), 2)}\r')

        time.sleep(5)

    @measurer
    def measure_filament_current(self):
        return float(self.query('FL?\r').decode().strip())

    @setter
    def set_mass(self, mass):
        self.write('ML' + f'{int(mass)}\r')

    @setter
    def set_masses(self, masses):

        initial_mass, final_mass = masses[0], masses[-1]

        self.write('MI' + f'{int(initial_mass)}\r')
        self.write('MF' + f'{int(final_mass)}\r')

    @setter
    def set_ppsf(self, value):
        self.write(f'SP{value}\r')

    @getter
    def get_ppsf(self):
        return float(self.query('SP?\r').decode().strip())

    @setter
    def set_tpsf(self, value):
        self.write(f'ST{value}\r')

    @getter
    def get_tpsf(self):
        return float(self.query('ST?\r').decode().strip())

    @measurer
    def measure_spectrum(self):

        self.write('HS1\r')

        bytes_waiting = 0
        while not bytes_waiting:
            bytes_waiting = self.adapter.backend.in_waiting
            time.sleep(self.adapter.delay)

        response = self.adapter.backend.read(bytes_waiting)

        return struct.unpack('<i', response)[0] * 1.0e-16 / self.ppsf

    @measurer
    def measure_single(self):

        self.write('MR'+f'{int(self.mass)}\r')

        bytes_waiting = 0
        while not bytes_waiting:
            bytes_waiting = self.adapter.backend.in_waiting
            time.sleep(self.adapter.delay)

        response = self.adapter.backend.read(bytes_waiting)

        return struct.unpack('<i', response)[0] * 1.0e-16 / self.ppsf

    @measurer
    def measure_total_pressure(self):

        self.write('TP?\r\n')

        bytes_waiting = 0
        while not bytes_waiting:
            bytes_waiting = self.adapter.backend.in_waiting
            time.sleep(self.adapter.delay)

        response = self.adapter.backend.read(bytes_waiting)

        return struct.unpack('<i', response)[0] * 1.0e-16 / self.tpsf
