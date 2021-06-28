# Extracting datasets from .RData file

* Install `R`
* In `R`:
    * `install.packages(c("DMwR"))`
    * `for (dsnr in 1:20) {write.csv(DSs[[dsnr]]@data, paste(DSs[[dsnr]]@name, ".csv", sep=""), row.names = F)}`
