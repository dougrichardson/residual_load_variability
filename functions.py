import xarray as xr
from scipy.stats import spearmanr

def load_monthly(name, convert="TWh", time_slice=slice("1941", None)):
    """
    Load monthly data.
    
    name: str, filename
    convert: str, 'GWh' or 'TWh' divides by 1e3 or 1e6, respectively
    time_slice: slice, times to select.
    """
    if convert == "GWh":
        divisor = 1000
    elif convert == "TWh":
        divisor = 1e6
    else:
        divisor = 1
        
    ds = xr.open_dataset("/g/data/w42/dr6273/work/projects/Aus_energy/monthly_data/" + name + ".nc")
    return ds.sel(time=time_slice) / divisor

# def load_add_mean(file, var, convert="TWh"):
#     """
#     Load monthly detrended data, then add mean of non-detrended and divide by 1000 (MWh to GWh).
    
#     file: str, filepath
#     var: str, name of variable
#     convert: str, 'GWh' or 'TWh' divides by 1e3 or 1e6, respectively
#     """
#     if convert == "GWh":
#         divisor = 1000
#     elif convert == "TWh":
#         divisor = 1e6
#     else:
#         divisor = 1
        
#     ds = load_monthly(file)
#     mean = ds[var].mean("time")
#     return (ds[var + "_detrended"] + mean) / divisor

def sel_month(ds, month):
    """
    Return array for specified month
    
    ds: dataset to select from
    month: int or list of int between 1 and 12, default is None
    """
    if month is None:
        return ds
    elif isinstance(month, int):
        if 1 <= month <= 12:
            return ds.isel(time=ds.time.dt.month == month)
        else:
            raise ValueError("Incorrect month specified.")
    elif isinstance(month, list):
        if all((isinstance(i, int)) & (1 <= i <= 12) for i in month):
            return ds.isel(time=ds.time.dt.month.isin(month))
        else:
            raise ValueError("Incorrect month specified.")

def detrend_dim(da, dim, deg=1):
    """
    Detrend along a single dimension.
    
    da: array to detrend
    dim: dimension along which to detrend
    deg: degree of polynomial to fit (1 for linear fit)
    
    Adapted from the original code here:
    Author: Ryan Abernathy
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def normalise(ds, groupby=None):
    """
    Return values with mean subtracted and divided by standard deviation.
    
    ds: dataset with 'time' dimension
    groupby: None, or str in form e.g. 'time.month'
    """
    if groupby is not None:
        return ds.groupby(groupby).apply(lambda x: (x - x.mean("time")) / x.std("time"))
    else:
        return (ds - ds.mean("time")) / ds.std("time")

def calc_contribution(ds, regions):
    """
    Percentage contribution of each region to the sum of all regions
    
    ds: dataset
    regions: list, regions to select from ds
    """
    cont = ds.sel(region=regions) / ds.sel(region=regions).sum("region") * 100
    return cont.rename("shortfall_contribution")

def xr_spearmanr(ds1, ds2):
    """
    xarray wrapper for scipt.stats.spearmanr
    """
    def _spearman(x, y):
        return spearmanr(x, y, nan_policy='omit')[0]
    
    return xr.apply_ufunc(
        _spearman,
        ds1,
        ds2,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[]],
        vectorize=True
    )