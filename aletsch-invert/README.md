
# Overview

This setup allows testing of the simplest optimization scheme for inverting an emulated ice flow model. The goal is to find the optimal ice thickness that best explains observational data while remaining consistent with the ice flow emulator, demonstrated here for the case of the Aletsch Glacier.

## Former set-up

- params_1.yaml : Opitmize thk to fit ice thickness and ice velocity
- params_2.yaml : Optimize thk, slidingco, usurf to fit ice thickness, ice velocity, usurf, ...

## Latest

Check at the separate branch igm-example-invert

# References (data)

@article{millan2019mapping,
  title={Mapping surface flow velocity of glaciers at regional scale using a multiple sensors approach},
  author={Millan, Romain and Mouginot, J{\'e}r{\'e}mie and Rabatel, Antoine and Jeong, Seongsu and Cusicanqui, Diego and Derkacheva, Anna and Chekki, Mondher},
  journal={Remote Sensing},
  volume={11},
  number={21},
  pages={2498},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}

@article{grab2021ice,
  title={Ice thickness distribution of all Swiss glaciers based on extended ground-penetrating radar data and glaciological modeling},
  author={Grab, Melchior and Mattea, Enrico and Bauder, Andreas and Huss, Matthias and Rabenstein, Lasse and Hodel, Elias and Linsbauer, Andreas and Langhammer, Lisbeth and Schmid, Lino and Church, Gregory and others},
  journal={Journal of Glaciology},
  volume={67},
  number={266},
  pages={1074--1092},
  year={2021},
  publisher={Cambridge University Press}
}

@article{linsbauer2021new,
  title={The new Swiss Glacier Inventory SGI2016: From a topographical to a glaciological dataset},
  author={Linsbauer, Andreas and Huss, Matthias and Hodel, Elias and Bauder, Andreas and Fischer, Mauro and Weidmann, Yvo and B{\"a}rtschi, Hans and Schmassmann, Emanuel},
  journal={Frontiers in Earth Science},
  pages={774},
  year={2021},
  publisher={Frontiers}
}


