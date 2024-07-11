#!/usr/bin/python
#
# Instance class
#

class Amodal(object):
    instID     = 0
    labelID    = 0
    pixelCount = 0
    medDist    = -1
    distConf   = 0.0

    def __init__(self, gtImageFile, imgNp, labelID, amodal_id):
        if (labelID == 255):
            return
        self.gtImageFile     = gtImageFile
        self.instID     = int(amodal_id.split('_')[-1])
        self.amodalID     = int(amodal_id.split('_')[0])
        self.labelID    = int(labelID)
        self.pixelCount = int(self.getInstancePixels(imgNp))


    def getInstancePixels(self, imgNp):
        return (imgNp != 0).sum()

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def toDict(self):
        buildDict = {}
        buildDict["gtImageFile"]     = self.gtImageFile
        buildDict["instID"]     = self.instID
        buildDict["labelID"]    = self.labelID
        buildDict["amodalID"]    = self.amodalID
        buildDict["pixelCount"] = self.pixelCount
        buildDict["medDist"]    = self.medDist
        buildDict["distConf"]   = self.distConf
        return buildDict

    def fromJSON(self, data):
        self.gtImageFile     = int(data["gtImageFile"])
        self.instID     = int(data["instID"])
        self.labelID    = int(data["labelID"])
        self.pixelCount = int(data["pixelCount"])
        if ("medDist" in data):
            self.medDist    = float(data["medDist"])
            self.distConf   = float(data["distConf"])

    def __str__(self):
        return "("+str(self.instID)+")"