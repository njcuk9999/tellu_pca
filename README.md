## Use of these codes

Note use of these codes will require a citation of two future papers:
1) the SPIRou DRS paper (in prep)
2) the paper explaning the SPIRou telluric procedure (Artigau 2019 - planned)

The SPIRou DRS is open source software and thus these codes are fully available to use by anyone.

## Some explanation of the SPIROU procedure

https://docs.google.com/presentation/d/134E89R0n7TDNUlqdEg4W-oCtBmHGZPjx_ET992XYjmw/edit#slide=id.g4cf00c220c_0_0


## How to use:

### setup for an instrument

need a setup for a specific instrument (see `setup_car.py` and `setup_spirou.py`) values will need modifying for each instrument. The most important is pushing the data into the correct format. For this one need two functions that are called in the main codes:
- `get_mk_tellu_data`
- `get_fit_tellu_data`

where the return should be the data added to `loc`:
```python
loc['MWAVE'] = mwave
loc['SFLUX'] = sflux
loc['BLAZE'] = sblaze
loc['SWAVE'] = swave
loc['TAPAS'] = tapas
loc['AIRMASS'] = airmass
```

where:
- `mwave` should be a 2D array that is of dimensions "number of orders" by "number of pixels in each order". The units should be in `nm` - this is the master wavelength solution which should not be changed once set (all absorption spectra outputs (`mk_tellu.py` outputs) will be shifted to this grid
- `sflux` should be a 2D array of the same dimensions as `mwave` this is the flux values for the telluric stars (for `mk_tellu.py` or for the science stars for `fit_tellu.py` - ideally `sflux` is not blaze corrected
- `blaze` should be a 2D array of the same dimensions, it should be the blaze correction for each pixel, if `sflux` is blaze corrected this should be a 2D array of 1s
- `swave` should be a 2D array of the same dimensions as `mwave` - it is the local wavelength solution of the telluric star (for `mk_tellu.py`) or the science (for `fit_tellu.py`)
- `tapas` is a astropy.table with the columns `wavelength`, `trans_h2o`, `trans_o3`, `trans_n2o`, `trans_o2`, `trans_co2`, `trans_ch4` -- where these are the transmission values from TAPAS, `wavelength` is in `nm`
- `airmass` is the airmass value from the telescope (usually the absolute airmass taken from the pointing)

As well as editing these one will need to edit `mk_tellu.py` and `fit_tellu.py` to add to the `if statement` options for different instruments.

### Running the code

- to make the absorption spectra (the database of tramissions needed for fitting the PCA) one can individually run `mk_tellu.py` or use the wrapper `mk_tellu_db.py` giving a list of files as appropriate. Where the file is set in the setup file or more conviently using the following code:
```python
import mk_tellu

mk_tellu.main('SPIROU', 'my_telluric_star.fits')
```

- to run the PCA fitting (after one has run a sufficient number of `mk_tellu.py` -- this number should be much greater than the number of PCA components -- one can run `fit_tellu.py` where the file is set in the setup file or  or more conviently using the following code:
```python
import fit_tellu

fit_tellu.main('SPIROU', 'my_science_star.fits')
```
