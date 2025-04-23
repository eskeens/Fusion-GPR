This work was supported in part by the U.S. Department of Energy, Office of Science, Office of Workforce Development for Teachers and Scientists (WDTS) under the Science Undergraduate Laboratory Internships Program (SULI).

This code was created for the purpose of fitting to the electron temperature and density profiles of the DIII-D tokamak discharge #162940, as presented in the cited work by Hassan et al.[1] and is designed to read the electron temperature and density p-files from that discharge. The original goal of this project was to generate fits to these profiles for the purpose of computationally reproducing microtearing mode (MTM) instabilities in the pedestal region of H-mode tokamaks.

The methods used and adapted in this code are outlined by Michoski et al.[2], with earlier versions of the code being adapted from peterroleants[3].

References:
1. E. Hassan, D. R. Hatch, M. R. Halfmoon, M. Curie, M. T. Kotchenreuther, S. M. Mahajan, G. Merlo, R. J. Groebner, A. O. Nelson, A. Diallo. "Identifying the microtearing modes in the pedestal of DIII-D H-modes using gyrokinetic simulations," Nucl. Fusion 62, 026008 (2021).
2. C. Michoski, T. A. Oliver, D. R. Hatch, A. Diallo, M. Kotschenreuther, D. Eldon, M. Waller, R. Groebner, A. O. Nelson. "A Gaussian process guide for signal regression in magnetic fusion." Nucl. Fusion 64, 035001 (2024).
3. peterroleants. "Gaussian Process (1/3) - From Scratch." https://peterroelants.github.io/posts/gaussian-process-tutorial/ (2019).
