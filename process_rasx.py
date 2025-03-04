# Author:       Hayden Robertson
# Date:         February 2025
# This script converts binary .rasx files from the Rigaku XRR to plain text files.
# Inspired by https://github.com/MaxBuchta/RASX-Python.

import os
import numpy as np
import zipfile
import io
from tkinter import filedialog
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import copy
import scipy


class XRR:
    """
    Parameters:
    ___________
    BeamHeight : float, height of the beam used for footprint correction calculations
    samewidth : float, sample width used for footprint correction calculations
    wavelength : float, wavelength of radiation. If not supplied, it will extract from the .rasx file
    background : float, default background to supply for background subtraction calculations.
    measurementCond : xml (bs4), instrument parameters from the acqusition of the first angular range

    """

    def __init__(
        self,
        BeamHeight=0.05,
        SampleWidth=10.0,
        BeamSmearing=0.01,
        SampleOffset=0,
        wavelength=None,
        bkg=None,
        measurementCond=None,
    ):
        self.BeamHeight = BeamHeight
        self.SampleWidth = SampleWidth
        self.BeamSmearing = BeamSmearing
        self.SampleOffset = SampleOffset
        self.bkg = bkg
        self.wavelength = wavelength
        self.measurementCond = measurementCond

    def footprint_corr(self, method="classic"):
        """
        Function to conduct a footprint correction on the data, assuning the shape of
        the beam adheres to two back-to-back error functions.
        """

        if method == "classic":
            _SampleWidth = self.SampleWidth
            _FWHM = self.BeamHeight
            _Sigma = self.BeamSmearing
            _SampleOffset = self.SampleOffset

            x = np.linspace(-1 * (_FWHM + 4 * _Sigma), (_FWHM + 4 * _Sigma), 10000)
            y = scipy.special.erf((_FWHM / 2 + x) / _Sigma) + scipy.special.erf(
                (_FWHM / 2 - x) / _Sigma
            )

            NormErf = sum(y)

            x0 = _SampleOffset

            _EffSampleHigh = _SampleWidth * np.sin(np.radians(self.theta))

            _FP = np.zeros_like(self.theta)

            for i in range(len(_FP)):
                mask = (-_EffSampleHigh[i] / 2 < (x - x0)) & ((x - x0) < _EffSampleHigh[i] / 2)
                _FP[i] = np.sum(y[mask])

            _FP[_FP == 0] = np.nan
            _FP /= NormErf
            _FP = 1 / _FP.T

            self.y *= _FP

        # ---- alt method -----
        if method == "alt":
            _new_y = copy.copy(self.y)
            for i, m in enumerate(_new_y):
                if self.SampleWidth * np.sin(self.theta[i]) >= self.BeamHeight:
                    _new_y[i] *= self.SampleWidth / self.BeamHeight * np.sin(self.theta[i])
                else:
                    continue
            self.y = _new_y

    def normalise(self):
        """
        Simple normalisation based on the maximum reflected itensity.
        """

        self.y /= max(self.y)

    def theta_to_q(self):
        """
        Converts the given theta values into q_z values.
        """

        _theta = self.theta / 2 * np.pi / 180
        q = 4 * np.pi * np.sin(_theta) / self.wavelength
        self.q = q
        self.x = q

    def process_data(self, file=None):
        """
        Function to import reflection data from a .rasx file.  This is the workhorse function.
        A file is imported and converted from the binary format. Each subfile is iterated across
        (typically containing different angular ranges) to provide a complete x, y dataset for
        the full theta range.

        Parameters:
        ____________
        file : string, default None. This is the .rasx file to be processed. If no file is provided,
        a open file dialogue box will appear.

        """

        times = []
        refls = []
        merged_x = []
        merged_y = []

        if file is None:
            file = filedialog.askopenfilename(
                title="Select a .rasx file",
                initialdir="/",
                filetypes=(
                    ("RASX", "*.rasx"),
                    ("All files", "*.*"),
                ),
            )
            self.pth = os.path.dirname(file)
            self.save_name = os.path.basename(file)[:-5]

        else:
            self.pth = "."
            self.save_name = os.path.basename(file)[:-5]

        with open(file, "rb") as binary_file:
            binary_data = binary_file.read()
            data = io.BytesIO(binary_data)

            with zipfile.ZipFile(data, "r") as zip_file:
                file_list = zip_file.namelist()

                if file_list:
                    for f in file_list:
                        if "Profile" in f:
                            with zip_file.open(f) as subfile:
                                subfile_content = subfile.read().decode("utf-8")
                                lines = subfile_content.split("\n")
                                data_columns = []
                                for line in lines:
                                    line = line.lstrip("\ufeff")
                                    columns = line.split("\t")
                                    numeric_columns = [
                                        float(column) for column in columns if column.strip()
                                    ]
                                    data_columns.append(numeric_columns)

                                data_columns = [
                                    (
                                        [
                                            x[0],
                                            x[1],
                                        ]
                                        if len(x) >= 3
                                        else x
                                    )
                                    for x in data_columns
                                ]
                                if data_columns and data_columns[-1] == []:
                                    data_columns.pop()

                                data_columns = np.array(data_columns)
                                data_columns[:, 0] = np.round(
                                    data_columns[:, 0],
                                    2,
                                )
                                data_columns[:, 1] = np.round(
                                    data_columns[:, 1],
                                    4,
                                )
                            refls.append(data_columns)

                        elif "MesurementConditions" in f:
                            with zip_file.open(f) as subfile:
                                subfile_content = subfile.read().decode("utf-8")

                            soup = BeautifulSoup(
                                subfile_content,
                                "xml",
                            )
                            t = float(soup.find("Speed").string)
                            times.append(t)

                            if self.wavelength is None:
                                self.wavelength = float(soup.find("WavelengthKalpha1").string)
                                self.measurementCond = soup

                        else:
                            continue

        for i, d in enumerate(refls):
            merged_y.append(refls[i][:, 1] / times[i])
            merged_x.append(refls[i][:, 0])

        merged_x = np.concatenate(merged_x)
        merged_y = np.concatenate(merged_y)

        self.x = merged_x
        self.theta = merged_x
        self.y = merged_y

    def background_corr(self):
        """
        Performs a background correction.

        If bkg is None, it will take 0.9 times the average of the last ten y points.
        If bkg is supplied, it will use that as a simple background subtraction.

        """
        if self.bkg is None:
            _background = np.average(self.y[-5:]) * 0.9
        else:
            _background = self.bkg

        self.y = self.y - _background

    def plot(self, xaxis="q"):
        """
        Plots the data

        Parameters:
        ___________
        xaxis : string, 'q' or 'theta' as xaxis options.
        """

        fig, ax = plt.subplots()

        if xaxis == "q":
            xlabel = "$q$, Å$^{-1}$"
            x = self.q
        elif xaxis == "theta":
            x = self.theta
            xlabel = "$\theta$, °"

        ax.scatter(x, self.y)

        ax.set_yscale("log")

        ax.set_ylabel("R")
        ax.set_xlabel(xlabel)

        return fig, ax

    def save_data(self, ask=True):
        """
        Saves the processed data as a text file.

        """

        if ask:

            _save_path = filedialog.asksaveasfilename(
                title="Select your save destination",
                initialdir=self.pth + "/",
                initialfile=self.save_name + ".dat",
                filetypes=(
                    ("DAT", "*.dat"),
                    ("TEXT", "*.txt"),
                    ("All files", "*.*"),
                ),
            )
        else:
            _save_path = self.pth + "/" + self.save_name + ".dat"

        try:
            x = self.q
            xlabel = "q"
        except:
            x = self.x
            xlabel = "theta"

        np.savetxt(
            _save_path,
            np.array([x, self.y]).T,
            header=f"{xlabel},R",
            delimiter=",",
            comments="",
        )
