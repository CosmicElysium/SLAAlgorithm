import threading
from scipy.optimize import curve_fit
from scipy.ndimage.filters import median_filter
import pyfits as fits
import numpy as np
from argos.util.decorator import synchronized, returns
from argos.lan.calibration.abstract_calibration_manager import \
    LatTargetPosition, LatLm1ReconstructionMatrix
from argos.exception import ArgosException
from argos.util.logger import Logger

FLEX_XPIX_PER_DEG = -1.158
FLEX_YPIX_PER_DEG = 0.449
MAX_LASERS = 6
SIGMA_FRAME = 4.5
MIN_CIRCLE_RADIUS = 60
MAX_CIRCLE_RADIUS = 80
GAUSS_WIDTH_GUESS = 5
FLUX_INT_WIDTH = 20
GAUSS_FIT_WIDTH = 40
LIGHT_SMOOTH_WINDOW = 5
LASER_RESOLUTION_SHALLOW = 5
LASER_RESOLUTION_STEEP = 10
LASER_SLOPE_TOLERANCE = 0.1
MINIMUM_LASER_POINTS = 5


class LatImageProcessError(ArgosException):
    pass


class WrongOrderException(ArgosException):
    def __init__(self, firstFunction, secondFunction):
        self.firstFunction = firstFunction
        self.secondFunction = secondFunction

    def __str__(self):
        return "Please run " + self.firstFunction +  \
            " before " + self.secondFunction


class NotEnoughLasersEx(ArgosException):
    pass


class NoLasersDetectedEx(ArgosException):
    pass


class BadGaussianFit(ArgosException):
    pass


class LatImageProcess(object):

    def __init__(self, calibrationManager):
        self._calibMgr = calibrationManager
        self._frameData = None
        self._darkSubtractedFrame = None
        self._noiseFilteredFrame = None
        self._height = None
        self._width = None
        self._lasers = []
        self._circle = None
        self._laserSlope = None
        self._minimumLaserFlux = None
        self._darkFrameTag = None
        self._flatFieldTag = None
        self._mutex = threading.RLock()
        self._logger = Logger.of('Lat Image Process')

    @synchronized("_mutex")
    def setDarkFrameTag(self, tag):
        assert isinstance(tag, str)
        self._darkFrameTag = tag

    @synchronized("_mutex")
    @returns(str)
    def darkFrameTag(self):
        return self._darkFrameTag

    def darkFrameAsNumPyArray(self):
        return self._calibMgr.loadLatDarkFrame(
            self._darkFrameTag).frameAsNumPyArray

    @synchronized("_mutex")
    def subtractDark(self):
        self._darkSubtractedFrame = \
            self._subtractImage(self.darkFrameAsNumPyArray())

    @synchronized("_mutex")
    def setNoiseFilteredFrame(self, frame):
        self._noiseFilteredFrame = frame

    @synchronized("_mutex")
    def noiseFilteredFrame(self):
        return self._noiseFilteredFrame

    def filterNoise(self):
        image = self._darkSubtractedFrame.copy()
        image[image < np.median(self._darkSubtractedFrame) + 3 * SIGMA_FRAME] = 0
        self.setNoiseFilteredFrame(median_filter(image, 3))

    @synchronized("_mutex")
    def setFlatFieldTag(self, tag):
        assert isinstance(tag, str)
        self._flatFieldTag = tag

    @synchronized("_mutex")
    @returns(str)
    def flatFieldTag(self):
        return self._flatFieldTag

    @synchronized("_mutex")
    def setLaserSlope(self, factor):
        self._laserSlope = factor

    @synchronized("_mutex")
    def getLaserSlope(self):
        return self._laserSlope

    @synchronized("_mutex")
    def setMinimumLaserFlux(self, counts):
        self._minimumLaserFlux = counts

    @synchronized("_mutex")
    def getMinimumLaserFlux(self):
        return self._minimumLaserFlux

    @synchronized("_mutex")
    def setFrameData(self, frameData):
        self._frameData = frameData

    @synchronized("_mutex")
    def frameData(self):
        return self._frameData

    @synchronized("_mutex")
    def setHeight(self, height):
        self._height = height

    @synchronized("_mutex")
    def setWidth(self, width):
        self._width = width

    @synchronized("_mutex")
    def height(self):
        return self._height

    @synchronized("_mutex")
    def width(self):
        return self._width

    def loadFrameFromVector(self, imageVector, height, width):
        self.setFrameData(imageVector.astype(np.float16))
        self.setHeight(height)
        self.setWidth(width)

    def loadFrameFromFits(self, frameFile):
        hdulist = fits.open(frameFile)
        fitsHead = hdulist[0].header
        self.setWidth(int(fitsHead['NAXIS1']))
        self.setHeight(int(fitsHead['NAXIS2']))
        self.setFrameData(hdulist[0].data.astype(np.float16))
        hdulist.close()

    def _subtractImage(self, image):
        return self.frameData() - image

    def grabRow(self, rowNumber):
        return np.copy(self.frameData()[rowNumber, :])

    def grabCol(self, colNumber):
        return np.copy(self.frameData()[:, colNumber])

    @synchronized("_mutex")
    def wipeLasers(self):
        self._lasers = []

    @synchronized("_mutex")
    def loadLaserList(self, laserList):
        self._lasers = laserList

    @synchronized("_mutex")
    def laserList(self):
        return self._lasers

    @synchronized("_mutex")
    def addLaser(self, laser):
        self._lasers.append(laser)

    def numberLaserPoints(self):
        return self._lasers[0].getNumberPoints()

    def findLasers(self, numberLasers):
        self.wipeLasers()
        self.subtractDark()
        self.filterNoise()
        if self.getLaserSlope() > 0:
            self._findLeftBaseBeamsGaussSubtraction(self.noiseFilteredFrame(),
                                                    numberLasers)
            if self.numberLaserPoints() < MINIMUM_LASER_POINTS:
                self.wipeLasers()
                self._findRightBaseBeamsGaussSubtraction(self.noiseFilteredFrame(),
                                                         numberLasers)
        else:
            self._findRightBaseBeamsGaussSubtraction(self.noiseFilteredFrame(),
                                                     numberLasers)
            if self.numberLaserPoints() < MINIMUM_LASER_POINTS:
                self.wipeLasers()
                self._findLeftBaseBeamsGaussSubtraction(self.noiseFilteredFrame(),
                                                        numberLasers)
        if self.numberLaserPoints() < MINIMUM_LASER_POINTS:
            NoLasersDetectedEx("No Lasers found!")
        for eachLaser in self.laserList():
            eachLaser.fitFinalLine(self.width(), self.height())
            if abs(self.getLaserSlope()) > 1:
                eachLaser.findHorizontalIntFluxProfile(self.noiseFilteredFrame())
            else:
                eachLaser.findVerticalIntFluxProfile(self.noiseFilteredFrame()) 
            eachLaser.smoothLight()
            eachLaser.findLaserEndPointsFromSmoothedIntLight()

    def _findRightBaseBeamsGaussSubtraction(self, image, numberLasers):
        tempLasers = []
        if abs(self.getLaserSlope()) >= 1:
            laserResolution = LASER_RESOLUTION_STEEP
        else:
            laserResolution = LASER_RESOLUTION_SHALLOW
        rowRange = range(self.height() - 1, -1, -laserResolution)
        for dummy in range(numberLasers):
            tempLasers.append(Laser())
        for row in rowRange:
            line = self.grabRow(row)
            try:
                gaussCandidates = Util.gaussianFinder(line, row,
                                                      self.getMinimumLaserFlux())
            except BadGaussianFit:
                continue
            if len(gaussCandidates) >= numberLasers:
                if (not tempLasers[0].getNumberPoints()) or \
                        Util.rightSidePeaksHaveValidSlopes(gaussCandidates,
                                                           tempLasers,
                                                           self.getLaserSlope()):
                    for i, eachLaser in enumerate(tempLasers):
                        eachLaser.appendPoint(gaussCandidates[-(i+1)])
                elif Util.leftSidePeaksHaveValidSlopes(gaussCandidates,
                                                       tempLasers[::-1],
                                                       self.getLaserSlope()):
                    for i, eachLaser in enumerate(reversed(self.laserList())):
                        eachLaser.appendPoint(gaussCandidates[i])
                elif tempLasers[0].getNumberPoints() == 1:
                    for eachLaser in tempLasers:
                        eachLaser.removeLastPoint()
        self.loadLaserList(tempLasers)

    def _findLeftBaseBeamsGaussSubtraction(self, image, numberLasers):
        height = self.height()
        tempLasers = []
        if abs(self.getLaserSlope()) >= 1:
            laserResolution = LASER_RESOLUTION_STEEP
        else:
            laserResolution = LASER_RESOLUTION_SHALLOW
        rowRange = range(height - 1, -1, -laserResolution)
        for dummy in range(numberLasers):
            tempLasers.append(Laser())
        for row in rowRange:
            line = self.grabRow(row)
            try:
                gaussCandidates = Util.gaussianFinder(line, row,
                                                      self.getMinimumLaserFlux())
            except BadGaussianFit:
                continue
            if len(gaussCandidates) >= numberLasers:
                if (not tempLasers[0].getNumberPoints()) or \
                        Util.leftSidePeaksHaveValidSlopes(gaussCandidates,
                                                          tempLasers,
                                                          self.getLaserSlope()):
                    for i, eachLaser in enumerate(tempLasers):
                        eachLaser.appendPoint(gaussCandidates[i])
                elif Util.rightSidePeaksHaveValidSlopes(gaussCandidates,
                                                        tempLasers[::-1],
                                                        self.getLaserSlope()):
                    for i, eachLaser in enumerate(reversed(tempLasers)):
                        eachLaser.appendPoint(gaussCandidates[-(i + 1)])
                elif tempLasers[0].getNumberPoints() == 1:
                    for eachLaser in tempLasers:
                        eachLaser.removeLastPoint()
        self.loadLaserList(tempLasers)

    def findCircle(self):
        self._circle = None
        laserSet = self._lasers
        if len(laserSet) == 3:
            self._circle = Circle()
            self._circle.findCircle(laserSet[0], laserSet[1], laserSet[2])
            self._checkCircleSanity(self._circle)

    def getCircle(self):
        return self._circle

    def _checkCircleSanity(self, circle):
        if self._isXOutsideOfFrame(circle.centerX):
            raise LatImageProcessError("Center X (%d) is out of bounds" % (
                                       circle.centerX))

        if self._isYOutsideOfFrame(circle.centerY):
            raise LatImageProcessError("Center Y (%d) is out of bounds" % (
                                       circle.centerY))

        if self._isRadiusUnrealistic(circle.radius):
            raise LatImageProcessError("Radius (%d) is unrealistic" % (
                                       circle.radius))

    def _isXOutsideOfFrame(self, x):
        return x < 0 and x > self._width

    def _isYOutsideOfFrame(self, y):
        return y < 0 and y > self._height

    def _isRadiusUnrealistic(self, radius):
        return radius < MIN_CIRCLE_RADIUS or radius > MAX_CIRCLE_RADIUS


class Circle(object):

    def __init__(self):
        self.radius = None
        self.centerX = None
        self.centerY = None

    def findCircle(self, laser1, laser2, laser3):
        circleParams = Util.findCircle(laser1.getEndPoint()[0],
                                       laser1.getEndPoint()[1],
                                       laser2.getEndPoint()[0],
                                       laser2.getEndPoint()[1],
                                       laser3.getEndPoint()[0],
                                       laser3.getEndPoint()[1])
        centerX = circleParams[0]
        centerY = circleParams[1]
        radius = circleParams[2]
        self.centerX = centerX
        self.centerY = centerY
        self.radius = radius


class LaserGaussian(object):
    def __init__(self, row, peakHeight, position, width, flux):
        self._row = row
        self._peakHeight = peakHeight
        self._position = position
        self._width = width
        self._flux = flux

    def getPeakHeight(self):
        return self._peakHeight

    def getRow(self):
        return self._row

    def getPosition(self):
        return self._position

    def getWidth(self):
        return self._width


class Laser(object):

    def __init__(self):
        self.xPoints = []
        self.yPoints = []
        self.peakHeightPoints = []
        self.lineX = None
        self.lineY = None
        self.path = None
        self.intFlux = None
        self.smoothedIntFlux = None
        self.pathEndIndex = None
        self.slope = None
        self.intercept = None
        self.invSlope = None

    def appendPoint(self, laserGaussian):
        self.xPoints.append(laserGaussian.getPosition())
        self.yPoints.append(laserGaussian.getRow())
        self.peakHeightPoints.append(laserGaussian.getPeakHeight())

    def removeLastPoint(self):
        self.xPoints.pop()
        self.yPoints.pop()

    def getNumberPoints(self):
        return len(self.xPoints)

    def fitFinalLine(self, width, height):
        try:
            params = np.polyfit(self.xPoints, self.yPoints, 1)
        except TypeError:
            raise LatImageProcessError("Cannot fit final laser line. Not enough points found.")
        self.slope = params[0]
        self.invSlope = -1 / self.slope
        self.intercept = params[1]
        self.lineX = np.arange(0, width, 1)
        self.lineY = (np.round(self.lineX *
                               self.slope + self.intercept)).astype(int)
        self.lineX = self.lineX[np.logical_and(self.lineY >= 0,
                                               self.lineY < height)]
        self.lineY = self.lineY[np.logical_and(self.lineY >= 0,
                                               self.lineY < height)]
        self.path = np.sqrt((self.lineX - self.lineX[0]) **
                            2 + (self.lineY - self.lineY[0]) ** 2)
        # Reverse laser coordinates if DX lasers
        if self.slope < 0:
            self.lineX = self.lineX[::-1]
            self.lineY = self.lineY[::-1]
            # PUT GEOMETRY HERE self.altitude =

    def findHorizontalIntFluxProfile(self, image):
        if self.path.size == 0:
            raise WrongOrderException("fitFinal_Line",
                                      "findIntFluxProfileAndEndPoint")
        self.intFlux = np.zeros(self.path.size)
        for xPixel, yPixel, index in zip(self.lineX, self.lineY,
                                         range(0, len(self.path))):
            xVals = np.arange(xPixel - FLUX_INT_WIDTH/2, xPixel +
                              FLUX_INT_WIDTH/2 + 1)
            xVals = xVals[np.logical_and(xVals >= 0, xVals < image.shape[1])]
            intLight = np.sum(image[yPixel, xVals])
            self.intFlux[index] = intLight
            if intLight == 0:
                break

    def findVerticalIntFluxProfile(self, image):
        if self.path.size == 0:
            raise WrongOrderException("fitFinal_Line",
                                      "findIntFluxProfileAndEndPoint")
        self.intFlux = np.zeros(self.path.size)
        for xPixel, yPixel, index in zip(self.lineX, self.lineY,
                                         range(0, len(self.path))):
            yVals = np.arange(yPixel - FLUX_INT_WIDTH/2, yPixel +
                              FLUX_INT_WIDTH/2 + 1)
            yVals = yVals[np.logical_and(yVals >= 0, yVals < image.shape[0])]
            intLight = np.sum(image[yVals, xPixel])
            self.intFlux[index] = intLight
            if intLight == 0:
                break

    def smoothLight(self):
        if np.sum(self.intFlux) == 0:
            raise NoLasersDetectedEx("No flux detected to smooth")
        self.smoothedIntFlux = \
            Util.movingaverage(np.asarray(self.intFlux), LIGHT_SMOOTH_WINDOW)

    def findLaserEndPointsFromSmoothedIntLight(self):
        try:
            self.pathEndIndex = np.where(self.smoothedIntFlux == 0)[0][0]
        except IndexError:
            raise LatImageProcessError("Lasers might leave CCD...")

    def getEndPoint(self):
        return np.array([self.lineX[self.pathEndIndex],
                         self.lineY[self.pathEndIndex]])


class Util(object):

    @staticmethod
    def movingaverage(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    @staticmethod
    def gaussian(x, a, sigma, x0):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    @staticmethod
    def findCircle(x1, y1, x2, y2, x3, y3):
        lineParamA = np.polyfit((x2, x1), (y2, y1), 1)
        lineParamB = np.polyfit((x2, x3), (y2, y3), 1)
        slopeA = lineParamA[0]
        slopeB = lineParamB[0]
        xCenter = (slopeA * slopeB * (y3 - y1) + slopeA * (x2 + x3) - slopeB *
                   (x1 + x2)) / (2 * (slopeA - slopeB))
        yCenter = -(xCenter - (x1 + x2) / 2) / slopeA + (y1 + y2) / 2
        rCenter = np.mean((np.sqrt((x1 - xCenter) ** 2 + (y1 - yCenter) ** 2),
                           np.sqrt((x2 - xCenter) ** 2 + (y2 - yCenter) ** 2),
                           np.sqrt((x3 - xCenter) ** 2 + (y3 - yCenter) ** 2)))
        return np.array((xCenter, yCenter, rCenter))

    @staticmethod
    def calculateSlope(x1, y1, x2, y2):
        return ((y2 - y1) / (x2 - x1))

    @staticmethod
    def leftSidePeaksHaveValidSlopes(listOfGaussians, laserList, desiredSlope):
        for index, eachLaser in enumerate(laserList):
            latest = [eachLaser.xPoints[-1], eachLaser.yPoints[-1]]
            slope = Util.calculateSlope(listOfGaussians[index].getPosition(),
                                        listOfGaussians[index].getRow(),
                                        latest[0], latest[1])
            if abs(slope - desiredSlope) > LASER_SLOPE_TOLERANCE:
                return False
        return True

    @staticmethod
    def rightSidePeaksHaveValidSlopes(listOfGaussians, laserList, desiredSlope):
        listOfGaussians = listOfGaussians[::-1]
        for index, eachLaser in enumerate(laserList):
            latest = [eachLaser.xPoints[-1], eachLaser.yPoints[-1]]
            slope = Util.calculateSlope(listOfGaussians[index].getPosition(),
                                        listOfGaussians[index].getRow(),
                                        latest[0], latest[1])
            if abs(slope - desiredSlope) > LASER_SLOPE_TOLERANCE:
                return False
        return True

    @staticmethod
    def gaussianFinder(line, row, fluxThreshold):
        maxEdge = len(line)
        foundGaussians = []
        foundPeaks = []
        foundGaussiansSorted = []
        for dummy in range(0, MAX_LASERS):
            if np.max(line) < fluxThreshold:
                break
            maxPos = line.argmax()
            maxVal = line[maxPos]
            fakeLine = np.zeros(maxEdge)
            lowBorder = maxPos - GAUSS_FIT_WIDTH/2
            if lowBorder < 0:
                lowBorder = 0
            highBorder = maxPos + GAUSS_FIT_WIDTH/2
            if highBorder > maxEdge:
                highBorder = maxEdge
            fakeLine[lowBorder:highBorder] = line[lowBorder:highBorder]
            initialParamGuess = [maxVal, GAUSS_WIDTH_GUESS, maxPos]
            rowPix = range(0, maxEdge)
            try:
                popt, pcov = curve_fit(Util.gaussian, rowPix,
                                       fakeLine, p0=initialParamGuess)
            except RuntimeError:
                print("Cannot find fit at pixel: " + str(maxPos) + "," +
                      str(row))
                raise BadGaussianFit
            height = popt[0]
            width = popt[1]
            position = popt[2]
            gFit = Util.gaussian(rowPix, height, width, position)
            fluxGauss = gFit.sum()
            foundGaussians.append(LaserGaussian(row, height,
                                                position, width, fluxGauss))
            foundPeaks.append(position)
            line -= gFit
        order = np.argsort(foundPeaks)
        for i in order:
            foundGaussiansSorted.append(foundGaussians[i])
        return foundGaussiansSorted
