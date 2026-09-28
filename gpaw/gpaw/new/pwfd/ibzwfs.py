from gpaw.new.ibzwfs import IBZWaveFunctions


class PWFDIBZWaveFunction(IBZWaveFunctions):
    def move(self, fracpos_ac, atomdist):
        super().move(fracpos_ac, atomdist)
        for wfs in self:
            wfs.move(fracpos_ac, atomdist)
