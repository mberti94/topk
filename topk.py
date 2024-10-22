#### TOPK code ####

#################### IMPORTANT ######################
### This section outlines the expected naming conventions for input files.
### 1) 21 cm Multipoles: Files should be named in the format "Pl_z" where z is the redshift and l is the multipole.
### EXAMPLE FOR MULTIPOLE: P0_025.dat is the name of the monopole data for redshift 0.25.
### EXAMPLE FOR MULTIPOLE: P2_175.dat is the name of the quadrupole data for redshift 1.75.
### 2) Covariance data (diagonal terms): Named in the format "covlk_z" where l,k are the multipoles, and z is the redshift.
### EXAMPLE FOR COVARIANCE: cov00_025.dat is the diagonal of the monopole-monopole covariance at z=0.25.
### EXAMPLE FOR COVARIANCE: cov02_175.dat is the monopole-quadrupole covariance at z=1.75.
### NOTE: l<k, so cov20_z.dat does not exist!!
### FOR CROSS: Same naming convention as for 21cm but P0_025.dat becomes --> PC0_025.dat, etc...
### FOR CROSS COVARIANCE: cov00_025.dat becomes --> covcross00_025.dat, etc...

import os 
import numpy as np  
import camb as my_camb  
from scipy.interpolate import interp1d 
from scipy.integrate import quad  
from scipy import integrate 
from typing import Optional, Sequence, List  
from cobaya.likelihood import Likelihood 
from cobaya.log import LoggedError 
from cobaya.conventions import Const, packages_path_input  

class topk(Likelihood):  # Inheriting from the Cobaya Likelihood base class
    # Class attributes with default values
    path: str = None  # Path to the directory containing data files
    cross_0: bool = False  # Indicates if cross monopole is included
    cross_2: bool = False  # Indicates if cross quadrupole is included
    topk_0: bool = False  # Indicates if 21cm monopole data is included
    topk_2: bool = False  # Indicates if 21cm quadrupole data is included
    topk_4: bool = False  # Indicates if 21cm hexadecapole data is included
    FoG: bool = False  # Finger-of-god effect; set to True for non-linear scales
    sigma_v: float = None  # Velocity dispersion for the FoG effect; requires further definition
    shot_noise: bool = False  # Indicates if shot noise is included; True for non-linear scales
    SN_as_nuis: bool = False  # Whether to treat shot noise as a nuisance parameter
    SN_nuis_fitted: bool = False  # Whether to fit shot noise as a nuisance parameter
    shot_noise_from_input: bool = False  # Whether to read shot noise from input files
    galaxy_bias_from_input: bool = False  # Whether to read galaxy bias from input files; still needs implementation
    nuisances_21: bool = False  # Indicates if nuisance parameters for 21cm data are used
    nuisances_cross: bool = False  # Indicates if nuisance parameters for cross data are used
    nuisances_fitted: bool = False  # Indicates if nuisance parameters are fitted
    nuis_quadratic_fit: bool = False  # Whether to use a quadratic fit for nuisance parameters
    gc_shot_noise: Optional[float] = None  # Galaxy clustering shot noise; implementation pending
    D: float = None  # Physical dimension of the telescope dish
    r_cross: Optional[float] = None  # Cross correlation parameter
    zs_21: Sequence[float] = None  # List of redshift values for the 21cm data
    zs_cross: Sequence[float] = None  # List of redshift values for cross data
    off_diag_cov: bool = None  # Whether to include off-diagonal terms in the covariance matrix
    AP_effect: bool = None  # Indicates if the Alcock-Paczynski effect is included
    T_b_model: str = None  # Model for the brightness temperature
    b_HI_model: str = None  # Model for the HI bias
    nonlinear_matter: bool = None  # Indicates if non-linear matter effects are considered
    H0_fid: float = 67.32  # Fiducial value for Hubble constant (H0)
    ombh2_fid: float = 0.022383  # Fiducial value for baryon density (ombh2)
    omch2_fid: float = 0.12011  # Fiducial value for cold dark matter density (omch2)
    tau_fid: float = 0.0543  # Fiducial value for optical depth (tau)
    mnu_fid: float = 0.06  # Fiducial value for neutrino mass [eV]

    params = {}  # Dictionary to hold model parameters

    def initialize(self):
        # Initialization checks
        if (self.topk_0 or self.topk_2 or self.topk_4) and self.zs_21 is None:
            raise LoggedError(self.log, "\nNeed to specify the redshift bins for the 21cm data")
        
        if (self.cross_0 or self.cross_2) and self.zs_cross is None:
            raise LoggedError(self.log, "\nNeed to specify the redshift bins for the cross data")
        
        if self.nuisances_21 and not (self.topk_0 or self.topk_2 or self.topk_4):
            raise LoggedError(self.log, "\nNuisances for 21cm cannot be switched on if no 21cm observables are selected.")
        
        if self.nuisances_cross and not (self.cross_0 or self.cross_2):
            raise LoggedError(self.log, "\nNuisances for cross cannot be switched on if no cross observables are selected.")
        
        # Setup for nuisance parameters
        if self.nuisances_21:
            if self.nuisances_fitted:
                if self.nuis_quadratic_fit:
                    # If fitting is required, set parameters for quadratic fitting
                    self.params.update({"a_21_1": None, "a_21_2": None, "b_21_1": None, "b_21_2": None, "c_21_1": None, "c_21_2": None})
                else:
                    # If not quadratic, setup parameters for linear fitting
                    self.params.update({"a_21_1": None, "a_21_2": None, "b_21_1": None, "b_21_2": None, "c_21_1": None, "c_21_2": None, "d_21_1": None, "d_21_2": None})
            else:
                # If not fitting nuisances, initialize default parameters for each redshift bin
                self.params.update({f'Tbsigma8_{i+1}': None for i in range(len(self.zs_21))})
                self.params.update({f'Tfsigma8_{i+1}': None for i in range(len(self.zs_21))})
                
        # Setup parameters for shot noise nuisance factors
        if self.SN_as_nuis:
            if self.SN_nuis_fitted:
                # If shot noise is fitted, setup parameters for it
                self.params.update({"a_sn": None, "b_sn": None, "c_sn": None, "d_sn": None})
            else:
                # If not fitting, initialize shot noise parameters for each redshift bin
                self.params.update({f'shot_noise_{i+1}': None for i in range(len(self.zs_21))})
              
        # Check if cross nuisances are enabled
        if self.nuisances_cross:
            # Check if nuisances for cross are fitted
            if self.nuisances_fitted:
                # If quadratic fit for nuisances is enabled
                if self.nuis_quadratic_fit:
                    # Update parameters for quadratic fit nuisances
                    self.params.update({"a_cross_1": None, "a_cross_2": None, "a_cross_3": None, 
                                        "b_cross_1": None, "b_cross_2": None, "b_cross_3": None, 
                                        "c_cross_1": None, "c_cross_2": None, "c_cross_3": None})
                else:
                    # Update parameters for non-quadratic fit nuisances
                    self.params.update({"a_cross_1": None, "a_cross_2": None, "a_cross_3": None, 
                                        "b_cross_1": None, "b_cross_2": None, "b_cross_3": None, 
                                        "c_cross_1": None, "c_cross_2": None, "c_cross_3": None, 
                                        "d_cross_1": None, "d_cross_2": None, "d_cross_3": None})
            else: 
                # Update parameters for nuisances when not fitted
                self.params.update({f'rTbsigma8_{i+1}': None for i in range(len(self.zs_cross))})
                self.params.update({f'rTbgsigma8_{i+1}': None for i in range(len(self.zs_cross))})
                self.params.update({f'rTfsigma8_{i+1}': None for i in range(len(self.zs_cross))})
        
        # Check if shot noise is provided from input and update parameters
        if self.shot_noise_from_input:
            self.params.update({f'shot_noise_input_{i+1}': None for i in range(len(self.zs_21))})

        # Check if galaxy bias is provided from input and update parameters
        if self.galaxy_bias_from_input:
            self.params.update({f'galaxy_bias_{i+1}': None for i in range(len(self.zs_cross))})

        # Define vectors for the 21cm and cross observables
        if (self.topk_0 or self.topk_2 or self.topk_4):
            self.observables_21 = []
            if self.topk_0:
                self.observables_21.append("topk_0")
            if self.topk_2:
                self.observables_21.append("topk_2")
            if self.topk_4:
                self.observables_21.append("topk_4")

        if (self.cross_0 or self.cross_2):
            self.observables_cross = []
            if self.cross_0:
                self.observables_cross.append("cross_0")
            if self.cross_2:
                self.observables_cross.append("cross_2")

        # Print selected observables for 21cm data
        if hasattr(self, 'observables_21') and self.observables_21:
            print(f"\nList of 21cm observables selected : {self.observables_21}.")
        else:
            print("\nNo 21cm observables selected.")

        # Print selected cross observables
        if hasattr(self, 'observables_cross') and self.observables_cross:
            print(f"\nList of cross observables selected : {self.observables_cross}.")
        else:
            print("\nNo cross observables selected.\n")
            
        # Ensure at least one observable is selected
        if not (self.topk_0 or self.topk_2 or self.topk_4 or self.cross_0 or self.cross_2):
            raise LoggedError(self.log, "\nNo observables selected. At least one of topk_0, topk_2, topk_4, cross_0, cross_2 must be True.\n")
         
        # Check for the path to data files
        if not getattr(self, "path", None) and \
                not getattr(self, packages_path_input, None):
            raise LoggedError(
                self.log, "\nNo path given to the data. Set the likelihood property "
                          "'path' or the common property '%s'.", packages_path_input)
        
        # If no path specified, use the external packages path
        data_file_path = os.path.normpath(getattr(self, "path", None) or
                                          os.path.join(self.packages_path, "data")) 
        
        # Check shot noise status
        if (self.topk_0 or self.topk_2 or self.topk_4):
            if self.shot_noise:    
                self.log.info("\nShot noise is set to True.\n")
            else:
                self.log.warning("\nShot noise is set to False. Shot noise should always be set to True for non-linear scales.\n")
                
        # Set default value for r_cross if not specified
        if (self.cross_0 or self.cross_2) and self.r_cross is None:
            self.log.warning("\nNo value for r_cross specified, r_cross = 1 by default.")
            self.r_cross = 1.
        
        # Validate T_b_model and b_HI_model if nuisances for 21cm are not enabled
        if not self.nuisances_21:
            if self.T_b_model not in ['Battye_13', 'Furlanetto_06']:
                self.log.warning(f"\nInvalid T_b_model: {self.T_b_model}. "
                                "Supported models are 'Battye_13' and 'Furlanetto_06', "
                                "model 'Battye_13' implemented by default.\n")
                self.T_b_model = 'Battye_13'
            
            if self.b_HI_model not in ['Navarro_18', 'Casas_23']:
                self.log.warning(f"\nInvalid b_HI_model: {self.b_HI_model}. "
                                "Supported models are 'Navarro_18' and 'Casas_23', "
                                "model 'Navarro_18' implemented by default.\n")
                self.b_HI_model = 'Navarro_18'

        # Check if the dish size D is specified; if not, set to default
        if self.D is None:
            self.log.warning("\nNo value for the dish size D, D = 15m by default\n")
            self.D = 15.   

        # Check if the FoG (Finger of God) effect is enabled and sigma_v is not specified
        if self.FoG and self.sigma_v is None:
            self.log.warning("\nNo value for sigma_v was specified, sigma_v = 200 km/s by default.")
            self.sigma_v = 200.  # km/s according to Soares (2020)
        elif not self.FoG:
            # If FoG is disabled, set sigma_v to zero
            self.log.warning("\nFoG is set to False, sigma_v = 0.")
            self.sigma_v = 0.
        
        # Ensure galaxy bias from input is only enabled if cross observables are selected
        if self.galaxy_bias_from_input and not (self.cross_0 or self.cross_2):
            raise LoggedError(self.log, "\nGalaxy bias from input cannot be switched on if no cross observables are selected.")

        # Check if the Alcock-Paczynski effect is specified; default to False if not
        if self.AP_effect is None:
            self.log.warning("\nNo specification for the Alcock-Paczynski effect, AP_effect = False by default.")
            self.AP_effect = False

        # Check if the matter power spectrum (PS) type is specified; default to nonlinear if not
        if self.nonlinear_matter is None:
            self.log.warning("\nNo specification for linear or nonlinear matter PS, nonlinear is set by default")
            self.nonlinear_matter = True

        # Check if off-diagonal terms in the covariance matrix are specified; default to True if not
        if self.off_diag_cov is None:
            self.log.warning("\nNo specification for off-diagonal terms in the covariance matrix, off_diag_cov = True by default.")
            self.off_diag_cov = True
            
        # Ensure shot noise as nuisance is enabled if shot noise is enabled
        if self.SN_as_nuis and not self.shot_noise: 
            self.log.warning("\nShot-noise as nuisance is switched on but shot-noise is switched-off, shot-noise turned on by default.")
            self.shot_noise = True
        
        # Check if shot noise from input is specified but currently disabled
        if self.shot_noise_from_input and not self.shot_noise: 
            self.log.warning("\nShot-noise to be read from input is switched on but shot-noise is switched-off, shot-noise turned on by default.")
            self.shot_noise = True
        
        # Raise an error if both shot noise from input and as nuisance are enabled
        if self.shot_noise_from_input and self.SN_as_nuis:
            raise LoggedError(self.log, "\nshot-noise from input and shot-noise as nuisance cannot be switched on at the same time.")
        
        if hasattr(self, 'zs_21') and self.zs_21:   # Check if self.zs_21 exists, is not empty, and has fewer than 5 elements
            if len(self.zs_21) < 5:
                lower_bound = max(self.zs_21[0] - 0.25, 0)  # Ensure lower bound is >= 0
                upper_bound = self.zs_21[-1] + 0.25
                zs_21_PS = np.linspace(lower_bound, upper_bound, 10)  # Create an array of redshifts for the PK interpolator
            else:
                zs_21_PS = np.array(self.zs_21)
            
        if hasattr(self, 'zs_cross') and self.zs_cross:  # Check if self.zs_cross exists, is not empty, and has fewer than 5 elements
            if len(self.zs_cross) < 5:
                lower_bound = max(self.zs_cross[0] - 0.25, 0)  # Ensure lower bound is >= 0
                upper_bound = self.zs_cross[-1] + 0.25
                zs_cross_PS = np.linspace(lower_bound, upper_bound, 10)  # Create an array of redshifts for the PK interpolator
            else: 
                zs_cross_PS = np.array(self.zs_cross)
                
        # Check conditions and perform operations
        if (self.topk_0 or self.topk_2 or self.topk_4) and (self.cross_0 or self.cross_2):
            # Case 1: Concatenate zs_21 and zs_cross with unique elements
            combined_zs = list(set(self.zs_21 + self.zs_cross))
            self.zs = np.sort(combined_zs)
            combined_zs_PK = np.unique(np.concatenate((zs_21_PS, zs_cross_PS)))
            self.zs_PK = np.sort(combined_zs_PK)
            print(f"\nRedshift bins for 21cm: {self.zs_21}\n")
            print(f"\nRedshift bins for cross: {self.zs_cross}\n")

        elif (self.topk_0 or self.topk_2 or self.topk_4) and not (self.cross_0 or self.cross_2):
            # Case 2: zs_21 becomes zs
            self.zs = np.array(self.zs_21)
            self.zs_PK = np.array(zs_21_PS)  
            print(f"\nRedshift bins for 21cm: {self.zs_21}\n")
        elif not (self.topk_0 or self.topk_2 or self.topk_4) and (self.cross_0 or self.cross_2):
            # Case 3: zs_cross becomes zs
            self.zs = np.array(self.zs_cross)
            self.zs_PK = np.array(zs_cross_PS)
            print(f"\nRedshift bins for cross: {self.zs_cross}\n")

        # Define a function to check and assign fiducial values for cosmological parameters
        def check_and_assign_fiducials(**variables):
            defaults = {
                'H0_fid': 67.32,  # Planck 2018
                'ombh2_fid': 0.022383,
                'omch2_fid': 0.12011,
                'tau_fid': 0.0543,
                'mnu_fid': 0.06}

            # Iterate through each variable and assign default if None
            for var_name, value in variables.items():
                if value is None:
                    self.log.warning(f"Warning: '{var_name}' is None. Assigning default value {defaults[var_name]}.")
                    variables[var_name] = defaults[var_name]
            
            return variables

        # Prepare parameters for checking and assigning fiducials
        pars = {
            'H0_fid': self.H0_fid,
            'ombh2_fid': self.ombh2_fid,
            'omch2_fid': self.omch2_fid,
            'tau_fid': self.tau_fid,
            'mnu_fid': self.mnu_fid}
        
        # Call the function to check and assign fiducial values
        pars = check_and_assign_fiducials(**pars)
        
                # Check if the Alcock-Paczynski effect is enabled
        if self.AP_effect: 
            # Create a CAMBparams object to set cosmological parameters
            parameters = my_camb.CAMBparams()
            parameters.set_cosmology(H0=pars['H0_fid'], ombh2=pars['ombh2_fid'], omch2=pars['omch2_fid'], tau=pars['tau_fid'], mnu=pars['mnu_fid'])
            # Obtain results from CAMB based on the specified parameters
            results = my_camb.get_results(parameters)

            # If cross observables are selected, compute Hubble parameter and angular diameter distance for cross redshifts
            if (self.cross_0 or self.cross_2):
                self.H_z_fid_cross = results.hubble_parameter(self.zs_cross)  # km/s/Mpc
                self.D_A_fid_cross = results.angular_diameter_distance(self.zs_cross)

            # If topk observables are selected, compute Hubble parameter and angular diameter distance for 21cm redshifts
            if (self.topk_0 or self.topk_2 or self.topk_4):
                self.H_z_fid_21 = results.hubble_parameter(self.zs_21)  # km/s/Mpc
                self.D_A_fid_21 = results.angular_diameter_distance(self.zs_21)

        # Print a recap of the parameters used in the run
        print(f"\nRecap of the parameters for the run:\n")
        
        # Check if 21cm observables are defined and print them
        if hasattr(self, 'observables_21') and self.observables_21:
            print(f"Observables switched on: \n21cm: {self.observables_21}\n")
        else:
            print("No 21cm observables selected.\n")

        # Check if cross observables are defined and print them
        if hasattr(self, 'observables_cross') and self.observables_cross:
            print(f"Cross observables selected: {self.observables_cross}\n")
        else:
            print("No cross observables selected.\n")

        # Print the redshift bins for 21cm if they are available
        if self.zs_21 is not None:
            print(f"Redshift bins for 21cm: {self.zs_21}\n")
        else:
            print("No redshift bins for 21cm.\n")
        
        # Print the redshift bins for cross observables if they are available
        if self.zs_cross is not None:
            print(f"Redshift bins for cross: {self.zs_cross}\n")
        else:
            print("No redshift bins for cross.\n")
            
        # Check and print nuisance parameters for both 21cm and cross observables
        if self.nuisances_21 or self.nuisances_cross:
            print(f"Nuisances 21: {self.nuisances_21}\n")
            print(f"Nuisances cross: {self.nuisances_cross}\n")
        else:
            print("No nuisance parameters.\n")
            print(f"21cm T_b model: {self.T_b_model}\n")
            print(f"21cm b_HI model: {self.b_HI_model}\n")
            
        # Check if any nuisance parameters are fitted and their fitting method
        if self.nuisances_fitted:
            if self.nuis_quadratic_fit:
                print("Nuisances to be fitted with a quadratic polynomial.\n")
            else: 
                print("Nuisances to be fitted with a cubic polynomial.\n")
        else: 
            print("No fitted nuisance parameters.\n")
        
        # Print cross parameter information based on the presence of nuisances
        if (self.cross_0 or self.cross_2) and not self.nuisances_cross:
            print(f"Cross parameter r: {self.r_cross}\n")
        else:
            print("No cross parameter r as nuisances are switched on.\n")
        
        # Determine if galaxy bias is sourced from input or model
        if (self.cross_0 or self.cross_2):
            if self.galaxy_bias_from_input:
                print("Galaxy bias to be read from input.\n")
            else:
                print("Galaxy bias computed from model.\n")  # Consider changing wording for clarity
        
        # Print shot noise information
        print(f"Shot noise: {self.shot_noise}\n")

        # Check shot noise treatment and fitting
        if self.SN_as_nuis: 
            if self.SN_nuis_fitted:
                print("Shot-noise to be treated as a nuisance parameter to be fitted in redshift.\n")
            else:
                print("Shot-noise to be treated as a nuisance parameter.\n")
                
        # Print if shot noise is read from input
        if self.shot_noise_from_input:
            print("Shot-noise to be read from input.\n")

        # Print other important parameters
        print(f"FoG: {self.FoG}\n")
        print(f"AP effect: {self.AP_effect}\n")
        print(f"Dish size: {self.D}\n")
        print(f"Non-linear matter power spectrum: {self.nonlinear_matter}\n")
        print(f"Off-diagonal terms in the covariance matrix: {self.off_diag_cov}\n")

        # Print fiducial values used in calculations
        print(f"Fiducial values: H0 = {pars['H0_fid']}, ombh2 = {pars['ombh2_fid']}, omch2 = {pars['omch2_fid']}, tau = {pars['tau_fid']}, mnu = {pars['mnu_fid']}\n")

    
        # We now load the data files ### READ ABOVE FOR NAMES###
        
        def load_col_1(file_name):
            # Construct the full file path by joining the data file path with the provided file name
            file_path = os.path.join(data_file_path, file_name)
            # Check if the file exists; raise an error if it doesn't
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"\nFile not found: {file_path}")
            # Load and return the first column of data from the file
            return np.loadtxt(file_path, usecols=0)

        def load_col_2(file_name):
            # Construct the full file path by joining the data file path with the provided file name
            file_path = os.path.join(data_file_path, file_name)
            # Check if the file exists; raise an error if it doesn't
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"\nFile not found: {file_path}")
            # Load and return the second column of data from the file
            return np.loadtxt(file_path, usecols=1)

        # Consider switching to gentxt format

        # Check if redshift bins for 21cm are defined
        if self.zs_21 is not None:
            self.z_21 = np.array(self.zs_21)  # Convert redshift bins to a NumPy array
            self.number_bins_21 = len(self.zs_21)  # Get the number of redshift bins
            self.k_21_data = []  # Initialize an empty list to hold k values (h Mpc^-1)

            # Check if topk_0 is enabled
            if self.topk_0:
                self.topk_0_data = []  # Initialize empty list for topk_0 data
                self.cov_21_00 = []  # Initialize empty list for covariance data

                # Loop over each redshift bin
                for i in range(self.number_bins_21):
                    # Create file names for data and covariance
                    self.file_21_0 = f"P0_{round(self.zs_21[i]*100):03d}.dat"
                    self.file_21_cov_00 = f"cov00_{round(self.zs_21[i]*100):03d}.dat"
                    try:
                        # Load data for k and topk_0, and store in respective lists
                        self.k_21_data.append(load_col_1(self.file_21_0))
                        self.topk_0_data.append(load_col_2(self.file_21_0))
                        self.cov_21_00.append(np.diag(load_col_1(self.file_21_cov_00)))  # Load diagonal covariance
                    except FileNotFoundError as e:
                        # Raise an error if file loading fails
                        raise LoggedError(self.log, f"Error loading file: {e}")

                print("\nTopk_0 files loading successful.")

            # Check if topk_2 is enabled
            if self.topk_2:
                self.topk_2_data = []  # Initialize empty list for topk_2 data
                self.cov_21_22 = []  # Initialize empty list for covariance data
                self.cov_21_02 = []  # Initialize empty list for off-diagonal covariance data

                # Loop over each redshift bin
                for i in range(self.number_bins_21):
                    # Create file names for data and covariance
                    self.file_21_2 = f"P2_{round(self.zs_21[i]*100):03d}.dat"
                    self.file_21_cov_02 = f"cov02_{round(self.zs_21[i]*100):03d}.dat"
                    self.file_21_cov_22 = f"cov22_{round(self.zs_21[i]*100):03d}.dat"
                    try:
                        # Load topk_2 data and covariance, and store in respective lists
                        self.topk_2_data.append(load_col_2(self.file_21_2)) 
                        self.cov_21_22.append(np.diag(load_col_1(self.file_21_cov_22))) 

                        # Check if off-diagonal covariance is enabled
                        if self.off_diag_cov:
                            self.cov_21_02.append(np.diag(load_col_1(self.file_21_cov_02)))
                        else:
                            # If not, append zeros matching the shape of the covariance
                            self.cov_21_02.append(np.zeros_like(self.cov_21_22[i]))  # Ensure consistency in dimensions

                    except FileNotFoundError as e:
                        # Raise an error if file loading fails
                        raise LoggedError(self.log, f"Error loading file: {e}")

                print("\nTopk_2 files loading successful.")

            # Check if topk_4 is enabled
            if self.topk_4:
                self.topk_4_data = []  # Initialize empty list for topk_4 data
                self.cov_21_04 = []  # Initialize empty list for covariance data
                self.cov_21_24 = []  # Initialize empty list for cross covariance data
                self.cov_21_44 = []  # Initialize empty list for covariance data

                # Loop over each redshift bin
                for i in range(self.number_bins_21):
                    # Create file names for data and covariance
                    self.file_21_4 = f"P4_{round(self.zs_21[i]*100):03d}.dat"
                    self.file_21_cov_04 = f"cov04_{round(self.zs_21[i]*100):03d}.dat"
                    self.file_21_cov_24 = f"cov24_{round(self.zs_21[i]*100):03d}.dat"
                    self.file_21_cov_44 = f"cov44_{round(self.zs_21[i]*100):03d}.dat"
                    try:
                        # Load topk_4 data and covariance, and store in respective lists
                        self.topk_4_data.append(load_col_2(self.file_21_4)) 
                        self.cov_21_44.append(np.diag(load_col_1(self.file_21_cov_44))) 

                        # Check if off-diagonal covariance is enabled
                        if self.off_diag_cov:
                            self.cov_21_04.append(np.diag(load_col_1(self.file_21_cov_04)))  
                            self.cov_21_24.append(np.diag(load_col_1(self.file_21_cov_24)))
                        else:    
                            # If not, append zeros matching the shape of the covariance
                            self.cov_21_04.append(np.zeros_like(self.cov_21_44[i]))
                            self.cov_21_24.append(np.zeros_like(self.cov_21_44[i]))

                    except FileNotFoundError as e:
                        # Raise an error if file loading fails
                        raise LoggedError(self.log, f"Error loading file: {e}")

                print("\nTopk_4 files loading successful.")

          
            # Initialize an empty list to store the inverted covariance matrices for 21cm data
            self.inv_cov_21 = []

            # Iterate over each redshift bin
            for i in range(self.number_bins_21):
                if self.topk_0 and not self.topk_2 and not self.topk_4:
                    # Case 1: Only topk_0 is True
                    inv_cov_00 = np.linalg.inv(self.cov_21_00[i])  # Invert the covariance matrix for topk_0
                    self.inv_cov_21.append(inv_cov_00)  # Append the inverted matrix to the list

                elif self.topk_0 and self.topk_2 and not self.topk_4:
                    # Case 2: topk_0 and topk_2 are True, topk_4 is False
                    cov_00 = self.cov_21_00[i]  # Covariance for topk_0
                    cov_02 = self.cov_21_02[i]  # Off-diagonal covariance between topk_0 and topk_2
                    cov_22 = self.cov_21_22[i]  # Covariance for topk_2
                    
                    # Create the top block combining cov_00 and cov_02
                    top_block = np.hstack((cov_00, cov_02))
                    # Create the bottom block combining cov_02 (transposed) and cov_22
                    bottom_block = np.hstack((cov_02.T, cov_22))
                    
                    # Combine the top and bottom blocks into a single covariance matrix
                    combined_cov = np.vstack((top_block, bottom_block))
                    inv_combined_cov = np.linalg.inv(combined_cov)  # Invert the combined covariance
                    
                    # Append the inverted matrix to the list
                    self.inv_cov_21.append(inv_combined_cov)

                elif self.topk_0 and self.topk_2 and self.topk_4:
                    # Case 3: topk_0, topk_2, and topk_4 are True
                    cov_00 = self.cov_21_00[i]  # Covariance for topk_0
                    cov_02 = self.cov_21_02[i]  # Off-diagonal covariance between topk_0 and topk_2
                    cov_04 = self.cov_21_04[i]  # Off-diagonal covariance between topk_0 and topk_4
                    cov_22 = self.cov_21_22[i]  # Covariance for topk_2
                    cov_24 = self.cov_21_24[i]  # Off-diagonal covariance between topk_2 and topk_4
                    cov_44 = self.cov_21_44[i]  # Covariance for topk_4
                    
                    # Create the top block combining cov_00, cov_02, and cov_04
                    top_block = np.hstack((cov_00, cov_02, cov_04))
                    # Create the middle block combining cov_02 (transposed), cov_22, and cov_24
                    middle_block = np.hstack((cov_02.T, cov_22, cov_24))
                    # Create the bottom block combining cov_04 (transposed), cov_24 (transposed), and cov_44
                    bottom_block = np.hstack((cov_04.T, cov_24.T, cov_44))
                    
                    # Combine the blocks into a single covariance matrix
                    combined_cov = np.vstack((top_block, middle_block, bottom_block))
                    inv_combined_cov = np.linalg.inv(combined_cov)  # Invert the combined covariance
                    
                    # Append the inverted matrix to the list
                    self.inv_cov_21.append(inv_combined_cov)

        # Check if cross-redshift bins are defined
        if self.zs_cross is not None:   
            self.z_cross = np.array(self.zs_cross)  # Convert cross-redshift bins to a NumPy array
            self.number_bins_cross = len(self.zs_cross)  # Get the number of cross-redshift bins
            self.k_cross_data = []  # Initialize an empty list for k values in cross data

            # Check if cross_0 is enabled
            if self.cross_0:
                self.cross_0_data = []  # Initialize empty list for cross_0 data
                self.cov_cross_00 = []  # Initialize empty list for covariance data

                # Loop over each cross-redshift bin
                for i in range(self.number_bins_cross):
                    # Create file names for cross data and covariance
                    self.file_cross_0 = f"PC0_{round(self.zs_cross[i]*100):03d}.dat"
                    self.file_cross_cov_00 = f"covcross00_{round(self.zs_cross[i]*100):03d}.dat"
                    try:
                        # Load k values and cross_0 data, and store in respective lists
                        self.k_cross_data.append(load_col_1(self.file_cross_0))
                        self.cross_0_data.append(load_col_2(self.file_cross_0))
                        self.cov_cross_00.append(np.diag(load_col_1(self.file_cross_cov_00)))  # Load diagonal covariance
                    except FileNotFoundError as e:
                        # Raise an error if file loading fails
                        raise LoggedError(self.log, f"Error loading file: {e}")
                        
                print("\nCross_0 files loading successful.")

            # Check if cross_2 is enabled
            if self.cross_2:
                self.cross_2_data = []  # Initialize empty list for cross_2 data
                self.cov_cross_02 = []  # Initialize empty list for off-diagonal covariance data
                self.cov_cross_22 = []  # Initialize empty list for covariance data

                # Loop over each cross-redshift bin
                for i in range(self.number_bins_cross):
                    # Create file names for cross data and covariance
                    self.file_cross_2 = f"PC2_{round(self.zs_cross[i]*100):03d}.dat"
                    self.file_cross_cov_02 = f"covcross02_{round(self.zs_cross[i]*100):03d}.dat"
                    self.file_cross_cov_22 = f"covcross22_{round(self.zs_cross[i]*100):03d}.dat"
                    try:
                        # Load cross_2 data and covariance, and store in respective lists
                        self.cross_2_data.append(load_col_2(self.file_cross_2))
                        self.cov_cross_22.append(np.diag(load_col_1(self.file_cross_cov_22)))  # Load diagonal covariance

                        # Check if off-diagonal covariance is enabled
                        if self.off_diag_cov:
                            self.cov_cross_02.append(np.diag(load_col_1(self.file_cross_cov_02)))  # Load off-diagonal covariance
                        else:
                            # If not, append zeros matching the shape of the covariance
                            self.cov_cross_02.append(np.zeros_like(self.cov_cross_22[i]))

                    except FileNotFoundError as e:
                        # Raise an error if file loading fails
                        raise LoggedError(self.log, f"Error loading file: {e}")
                        
                print("\nCross_2 files loading successful.")


            # Initialize an empty list to store the inverted covariance matrices for cross data
            self.inv_cov_cross = []

            # Iterate over each redshift bin
            for i in range(self.number_bins_cross):

                if self.cross_0 and not self.cross_2:
                    # Case 1: Only cross_0 is True
                    inv_cov_00 = np.linalg.inv(self.cov_cross_00[i])  # Invert the covariance matrix for cross_0
                    self.inv_cov_cross.append(inv_cov_00)  # Append the inverted matrix to the list

                elif self.cross_0 and self.cross_2:
                    # Case 2: Both cross_0 and cross_2 are True
                    cov_00 = self.cov_cross_00[i]  # Covariance matrix for cross_0
                    cov_02 = self.cov_cross_02[i]  # Off-diagonal covariance between cross_0 and cross_2
                    cov_22 = self.cov_cross_22[i]  # Covariance matrix for cross_2
                    
                    # Create the top block combining cov_00 and cov_02
                    top_block = np.hstack((cov_00, cov_02))
                    # Create the bottom block combining cov_02 (transposed) and cov_22
                    bottom_block = np.hstack((cov_02.T, cov_22))
                    
                    # Combine the top and bottom blocks into a single covariance matrix
                    combined_cov = np.vstack((top_block, bottom_block))
                    inv_combined_cov = np.linalg.inv(combined_cov)  # Invert the combined covariance
                    
                    # Append the inverted matrix to the list
                    self.inv_cov_cross.append(inv_combined_cov)

    # Define a method to get the requirements for various parameters
    def get_requirements(self):
        return {
            "H0": None,  # Hubble constant (can be set to None for default)
            "omegam": None,  # Matter density parameter (can be set to None for default)
            "ombh2": None,  # Baryon density parameter (can be set to None for default)
            "Pk_interpolator": {"z": self.zs_PK, "nonlinear": self.nonlinear_matter, "k_max": 10},  # Requirements for the power spectrum interpolator
            "Hubble": {"z": self.zs},  # Hubble parameter as a function of redshift
            "angular_diameter_distance": {"z": self.zs},  # Angular diameter distance as a function of redshift
            "sigma8_z": {"z": self.zs},  # Sigma8 at different redshifts
            "fsigma8": {"z": self.zs},  # Growth rate f*sigma8 at different redshifts
            "comoving_radial_distance": {"z": self.zs}  # Comoving radial distance as a function of redshift
        }

    # Now start the computation of the needed quantities
    def T_b(self, z, H0, H_z, ommh2, ombh2):
        # Compute the brightness temperature T_b at redshift z
        Omega_HI = 4. * 1.e-4 * (1. + z)**0.6  # Calculate the HI density using the formula from Crighton (2015)
        h = H0 / 100.  # Normalize H0 to dimensionless h
        omb = ombh2 / (h**2)  # Compute baryon density parameter

        # Check the model for T_b calculation and return the corresponding value
        if self.T_b_model == 'Battye_13':
            return 44. * 1.e-3 * ((Omega_HI * h) / (2.45 * 1.e-4)) * ((((1. + z)**2) * H0) / H_z)  # mK

        elif self.T_b_model == 'Furlanetto_06':
            return 23.88 * (ombh2 / 0.02) * np.sqrt((0.15 * (1. + z)) / (ommh2 * 10.)) * (Omega_HI / (0.74 * omb))  # mK

    def b_HI(self, z):
        # Compute the bias factor for HI at redshift z
        if self.b_HI_model == 'Navarro_18':
            b_known = [0.84, 1.49, 2.03, 2.56, 2.82, 3.18]  # Bias values from Navarro 2018
            z_known = np.arange(6)  # Known redshifts
            bHI_z = interp1d(z_known, b_known, kind='cubic')  # Interpolate bias values
            return bHI_z(z)  # Return interpolated value for redshift z

        elif self.b_HI_model == 'Casas_23':
            return 0.3 * (1. + z) + 0.6  # Alternative model for bias factor

    def nuis_cubic(self, z, a, b, c, d):
        # Compute a cubic nuisance function for redshift z
        return a * z**3 + b * z**2 + c * z + d

    def nuis_quadratic(self, z, a, b, c):
        # Compute a quadratic nuisance function for redshift z
        return a * z**2 + b * z + c 

    def b_g(self, z):
        # Compute the bias factor for galaxies at redshift z
        return np.sqrt(1. + z)  # Simple model of galaxy bias

    # Method to compute the full width at half maximum (FWHM) angle
    def theta_FWHM(self, z, D):
        """
        Calculate the FWHM in radians based on redshift z and diameter D.
        Input D is in meters.
        Output is in radians.
        """ 
        lambda_21 = 21.0  # Wavelength in cm for 21 cm line
        return 1.22 * lambda_21 * (1. + z) / (D * 100)  # Compute and return FWHM angle

    # Calculate the beam radius in Mpc h^-1
    def R_beam(self, sigma, r, h):
        """
        Calculate the beam radius based on sigma, radius r, and h.
        Input D is in meters.
        Output is in Mpc h^-1.
        """     
        return sigma * r * h  # Return calculated beam radius

    # Define q and nu, variables for the Alcock-Paczynski (AP) effect, k in h/Mpc units
    def q(self, k, mu, a_perp, a_par):
        # Compute q factor based on k, mu, a_perp, and a_par
        return (k / a_perp) * np.sqrt(1. + mu**2 * (a_perp**2 / a_par**2 - 1.))  # Return computed q

    def nu(self, mu, a_perp, a_par):
        # Compute nu factor based on mu, a_perp, and a_par
        return (a_perp * mu) / (a_par * np.sqrt(1. + mu**2 * (a_perp**2 / a_par**2 - 1.)))  # Return computed nu

    def SN_21(self, z, H0, H_z):  
        # Calculate the signal-to-noise ratio for 21 cm signal at redshift z
        PSN_known = [104, 124, 65, 39, 14, 7]  # Known PSN values from Navarro 2018
        z_known = np.arange(6)  # Known redshifts
        PSN_z = interp1d(z_known, PSN_known, kind='cubic')  # Interpolate PSN values
        Omega_HI = 4. * 1.e-4 * (1. + z)**0.6  # HI density calculation
        h = H0 / 100.  # Normalize H0
        T_b = 189 * h * Omega_HI * (((1. + z)**2) * H0 / H_z)  # Calculate brightness temperature
        P_SN = PSN_z(z) * (T_b)**2  # Compute PSN based on T_b
        return P_SN  # Return the computed signal-to-noise ratio

    def PS_21(self, mu, z, k, a_perp, a_par, R_beam, H0, P_m, nuisances=False, **kwargs): 
        # Calculate the power spectrum for 21 cm signal
        qq = self.q(k, mu, a_perp, a_par)  # Calculate q
        nn = self.nu(mu, a_perp, a_par)  # Calculate nu
        h = H0 / 100.  # Normalize H0
        B2 = np.exp(-(qq**2) * (R_beam**2) * (1. - (nn**2)))  # Compute beam factor
        FoG = 1. / (1. + (qq * nn * self.sigma_v / H0)**2)  # Finger-of-god factor
        
        P_SN = kwargs.get('P_SN', None)  # Get PSN from keyword arguments
        
        # Check if nuisance parameters are included
        if nuisances: 
            sigma_8 = kwargs.get('sigma_8', None)
            nuis_1 = kwargs.get('nuis_1', None)
            nuis_2 = kwargs.get('nuis_2', None)
            P =  FoG * B2 * ((nuis_1 + nuis_2 * nn**2)**2 * h**3 * (P_m.P(z, h * qq) / (sigma_8**2)) + P_SN) 
        else:   
            T_b = kwargs.get('T_b', None)
            b_HI = kwargs.get('b_HI', None)
            f = kwargs.get('f', None)
            P = FoG * B2 * ((T_b**2) * (((b_HI + f * nn**2)**2) * h**3 * P_m.P(z, h * qq)) + P_SN)  # Calculate power spectrum without nuisances
        return P  # Return computed power spectrum

    def PS_cross(self, mu, z, k, a_perp, a_par, R_beam, H0, P_m, nuisances=False, **kwargs): 
        # Calculate the cross-power spectrum
        qq = self.q(k, mu, a_perp, a_par)  # Calculate q
        nn = self.nu(mu, a_perp, a_par)  # Calculate nu
        h = H0 / 100.  # Normalize H0
        B = np.exp(-0.5 * (qq**2) * (R_beam**2) * (1. - (nn**2)))  # Compute beam factor
        FoG = 1. / (1. + (qq * nn * self.sigma_v / H0)**2)  # Finger-of-god factor

        # Check if nuisance parameters are included
        if nuisances: 
            sigma_8 = kwargs.get('sigma_8', None)
            nuis_1 = kwargs.get('nuis_1', None)
            nuis_2 = kwargs.get('nuis_2', None)
            nuis_3 = kwargs.get('nuis_3', None)
            P =  B * FoG * (nuis_1 + nuis_2 * nn**2) * (nuis_3 + nuis_2 * nn**2) * h**3 * (P_m.P(z, h * qq) / (sigma_8**2))  # Calculate power spectrum with nuisances
        else:   
            T_b = kwargs.get('T_b', None)
            b_g = kwargs.get('b_g', None)
            b_HI = kwargs.get('b_HI', None)
            f = kwargs.get('f', None)
            P = T_b * B * FoG * self.r_cross * (b_HI + f * nn**2) * (b_g + f * nn**2) * h**3 * P_m.P(z, h * qq)  # Calculate cross-power spectrum without nuisances

        return P  # Return computed cross-power spectrum

    # Function to define a wrapper for power spectra that multiplies them by Legendre polynomials for integration
    def P_to_integrate(self, P, mode):
        def wrapper(mu, *args, **kwargs):
            # Depending on the mode, apply different Legendre polynomial multipliers to the power spectrum
            if mode in ["topk_0", "cross_0"]:
                return P(mu, *args, **kwargs)  # No multiplier for 0th order
            
            elif mode in ["topk_2", "cross_2"]:
                # Apply 2nd-order polynomial: 5/2 * (3 * mu^2 - 1)
                return 5. * 0.5 * (3. * mu**2 - 1.) * P(mu, *args, **kwargs)

            elif mode == "topk_4":
                # Apply 4th-order polynomial: 9/8 * (35 * mu^4 - 30 * mu^2 + 3)
                return 9. * (35. * mu**4 - 30. * mu**2 + 3.) / 8. * P(mu, *args, **kwargs)

            else:
                # Raise an error for invalid mode input
                raise ValueError("Invalid multipole selected.")
        
        return wrapper

    # Function to integrate power spectra for the 21 cm signal based on given parameters
    def P_21_integration(self, mode, z, k, a_perp, a_par, R_beam, H0, P_m, nuisances=False, **kwargs):
        n_bins = len(z)  # Number of redshift bins
        P_SN = kwargs.get('P_SN', None)  # Retrieve the signal-to-noise ratio if provided
        integral = []  # List to hold the results of the integration

        # Iterate over each redshift bin
        for j in range(n_bins):
            n = len(k[j])  # Number of k values for the current redshift bin
            aux = []  # Temporary list to hold integrals for the current bin

            # Iterate over each k value
            for i in range(n):
                # Determine if nuisance parameters should be included in the calculations
                if nuisances:
                    sigma_8 = kwargs.get('sigma_8', None)
                    nuis_1 = kwargs.get('nuis_1', None) 
                    nuis_2 = kwargs.get('nuis_2', None) 

                    # Create a lambda function to integrate with nuisance parameters
                    P_lambda_func = lambda mu, *args: self.P_to_integrate(self.PS_21, mode)(
                        mu, *args, nuisances=nuisances, sigma_8=sigma_8[j], nuis_1=nuis_1[j], 
                        nuis_2=nuis_2[j], P_SN=P_SN[j]
                    )
                else:
                    # Create a lambda function to integrate without nuisance parameters
                    T_b = kwargs.get('T_b', None) 
                    b_HI = kwargs.get('b_HI', None)
                    f = kwargs.get('f', None) 

                    P_lambda_func = lambda mu, *args: self.P_to_integrate(self.PS_21, mode)(
                        mu, *args, nuisances=nuisances, T_b=T_b[j], b_HI=b_HI[j], f=f[j], P_SN=P_SN[j]
                    )
                
                # Perform the integration over mu from 0 to 1
                integ, _ = integrate.quad(P_lambda_func, 0., 1., args=(z[j], k[j][i], a_perp[j], a_par[j], R_beam[j], H0, P_m))
                aux.append(np.array(integ))  # Append the result to the temporary list
            
            # Normalize the results by dividing by the areas in a_perp and a_par
            integral.append(np.array(aux) / a_perp[j]**2 / a_par[j])

        return integral  # Return the accumulated integrals

    # Function to integrate the cross-power spectrum based on given parameters
    def P_cross_integration(self, mode, z, k, a_perp, a_par, R_beam, H0, P_m, nuisances=False, **kwargs):
        n_bins = len(z)  # Number of redshift bins
        integral = []  # List to hold the results of the integration

        # Iterate over each redshift bin
        for j in range(n_bins):
            n = len(k[j])  # Number of k values for the current redshift bin
            aux = []  # Temporary list to hold integrals for the current bin

            # Iterate over each k value
            for i in range(n):
                # Determine if nuisance parameters should be included in the calculations
                if nuisances:
                    sigma_8 = kwargs.get('sigma_8', None)
                    nuis_1 = kwargs.get('nuis_1', None)
                    nuis_2 = kwargs.get('nuis_2', None)
                    nuis_3 = kwargs.get('nuis_3', None)

                    # Create a lambda function to integrate with nuisance parameters
                    P_lambda_func = lambda mu, *args: self.P_to_integrate(self.PS_cross, mode)(
                        mu, *args, nuisances=nuisances, sigma_8=sigma_8[j], 
                        nuis_1=nuis_1[j], nuis_2=nuis_2[j], nuis_3=nuis_3[j]
                    )
                else:
                    # Create a lambda function to integrate without nuisance parameters
                    T_b = kwargs.get('T_b', None)
                    b_g = kwargs.get('b_g', None)
                    b_HI = kwargs.get('b_HI', None)
                    f = kwargs.get('f', None)

                    P_lambda_func = lambda mu, *args: self.P_to_integrate(self.PS_cross, mode)(
                        mu, *args, nuisances=nuisances, T_b=T_b[j], b_g=b_g[j], 
                        b_HI=b_HI[j], f=f[j]
                    )
                
                # Perform the integration over mu from 0 to 1
                integ, _ = integrate.quad(P_lambda_func, 0., 1., args=(z[j], k[j][i], a_perp[j], a_par[j], R_beam[j], H0, P_m))
                aux.append(np.array(integ))  # Append the result to the temporary list
            
            # Normalize the results by dividing by the areas in a_perp and a_par
            integral.append(np.array(aux) / a_perp[j]**2 / a_par[j])

        return integral  # Return the accumulated integrals

    # Function to retrieve data based on the specified observable
    def get_data(self, observable):
        attribute_name = observable + '_data'  # Construct the attribute name for the observable data
        if hasattr(self, attribute_name):
            return getattr(self, attribute_name)  # Return the data if it exists
        else:
            return None  # Return None if the data attribute does not exist

        
    # Function to compute theoretical predictions for a given observable
    def get_theory(self, observable, zs, nuisances_21=False, nuisances_cross=False, **kwargs): 
        # Retrieve cosmological parameters from the provider
        H0 = self.provider.get_param("H0")  # Hubble constant
        h2 = (H0 / 100.)**2  # h squared
        h = H0 / 100  # Dimensionless Hubble parameter
        omegam = self.provider.get_param("omegam")  # Matter density parameter
        ommh2 = omegam * h2  # Matter density in units of h^2
        ombh2 = self.provider.get_param("ombh2")  # Baryon density in units of h^2
        rs = self.provider.get_comoving_radial_distance(z=zs)  # Comoving radial distance for redshift zs
        P_m = self.provider.get_Pk_interpolator(nonlinear=self.nonlinear_matter)  # Power spectrum interpolator
        fsigma_8 = self.provider.get_fsigma8(z=zs)  # Growth rate times sigma8
        sigma8_z = self.provider.get_sigma8_z(z=zs)  # Sigma8 at redshift zs
        D_A = self.provider.get_angular_diameter_distance(z=zs)  # Angular diameter distance for redshift zs
        H_z = self.provider.get_Hubble(z=zs, units="km/s/Mpc")  # Hubble parameter at redshift zs
        sigma = self.theta_FWHM(zs, self.D) / 2. / np.sqrt(2. * np.log(2.))  # Beam width in radians
        R_beam = self.R_beam(sigma, rs, H0 / 100.)  # Beam radius based on the width and comoving distance

        # Check if the observable is a 21 cm power spectrum multipole
        if observable in ["topk_0", "topk_2", "topk_4"]:
            P_SN = kwargs.get('P_SN', None)  # Get signal-to-noise ratio if provided

            # Adjust for Alcock-Paczynski effect if applicable
            if self.AP_effect:
                a_par = self.H_z_fid_21 / H_z  # Scale factor in the line-of-sight direction
                a_perp = D_A / self.D_A_fid_21  # Scale factor in the transverse direction
            else: 
                # No Alcock-Paczynski effect, set scale factors to unity
                a_par = np.ones_like(self.zs_21)
                a_perp = np.ones_like(self.zs_21)

            # Check if nuisance parameters for 21 cm are included
            if nuisances_21: 
                nuis_1 = kwargs.get('nuis_1', None)  # Get first nuisance parameter if provided
                nuis_2 = kwargs.get('nuis_2', None)  # Get second nuisance parameter if provided
                # Perform the integration for the 21 cm signal with nuisance parameters
                multipoles = self.P_21_integration(
                    observable, zs, self.k_21_data, a_perp, a_par, R_beam, H0, P_m,
                    nuisances=nuisances_21, sigma_8=sigma8_z, nuis_1=nuis_1, nuis_2=nuis_2, P_SN=P_SN
                )  
            else:
                # Compute quantities needed for 21 cm signal without nuisance parameters
                T_b = self.T_b(zs, H0, H_z, ommh2, ombh2)  # Brightness temperature
                f = fsigma_8 / sigma8_z  # Growth rate
                b_HI = self.b_HI(zs)  # HI bias
                # Perform the integration for the 21 cm signal without nuisance parameters
                multipoles = self.P_21_integration(
                    observable, zs, self.k_21_data, a_perp, a_par, R_beam, H0, P_m,
                    nuisances=nuisances_21, T_b=T_b, f=f, b_HI=b_HI, P_SN=P_SN
                )
            
            return multipoles  # Return the computed multipoles

        # Check if the observable is a cross-power spectrum multipole
        if observable in ["cross_0", "cross_2"]:
            # Adjust for Alcock-Paczynski effect if applicable
            if self.AP_effect:
                a_par = self.H_z_fid_cross / H_z  # Scale factor in the line-of-sight direction
                a_perp = D_A / self.D_A_fid_cross  # Scale factor in the transverse direction
            else:
                # No Alcock-Paczynski effect, set scale factors to unity
                a_par = np.ones_like(self.zs_cross)
                a_perp = np.ones_like(self.zs_cross)

            # Check if nuisance parameters for cross-power spectrum are included
            if nuisances_cross:
                nuis_1 = kwargs.get('nuis_1', None)  # Get first nuisance parameter if provided
                nuis_2 = kwargs.get('nuis_2', None)  # Get second nuisance parameter if provided
                nuis_3 = kwargs.get('nuis_3', None)  # Get third nuisance parameter if provided
                # Perform the integration for the cross-power spectrum with nuisance parameters
                multipoles = self.P_cross_integration(
                    observable, zs, self.k_cross_data, a_perp, a_par, R_beam, H0, P_m,
                    nuisances=nuisances_cross, sigma_8=sigma8_z, nuis_1=nuis_1, nuis_2=nuis_2, nuis_3=nuis_3
                )
            else:
                # Compute quantities needed for cross-power spectrum without nuisance parameters
                T_b = self.T_b(zs, H0, H_z, ommh2, ombh2)  # Brightness temperature
                f = fsigma_8 / sigma8_z  # Growth rate
                b_HI = self.b_HI(zs)  # HI bias
                b_g = kwargs.get('b_g', None)  # Get galaxy bias if provided
                # Perform the integration for the cross-power spectrum without nuisance parameters
                multipoles = self.P_cross_integration(
                    observable, zs, self.k_cross_data, a_perp, a_par, R_beam, H0, P_m,
                    nuisances=nuisances_cross, T_b=T_b, f=f, b_HI=b_HI, b_g=b_g
                )
            
            return multipoles  # Return the computed multipoles

    def logp(self, **params_values): 
        # Initialize log probability results for cross and 21 cm observables
        result_cross = 0.
        result_21 = 0.  

        # Check if the object has 21 cm observables
        if hasattr(self, 'observables_21') and self.observables_21:
            theory_21_aux = []  # To store theoretical values for 21 cm observables
            data_21_aux = []    # To store data values for 21 cm observables
            self.data_21 = []   # Final data for 21 cm observables
            self.theory_21 = [] # Final theoretical predictions for 21 cm observables

            # Handle shot noise based on specified conditions
            if self.shot_noise:
                if self.SN_as_nuis:  # If shot noise is treated as a nuisance parameter
                    if self.SN_nuis_fitted:  # Check if shot noise parameters are fitted
                        # Extract shot noise coefficients from params_values
                        a_sn = params_values['a_sn']
                        b_sn = params_values['b_sn']
                        c_sn = params_values['c_sn']
                        d_sn = params_values['d_sn']
                        # Calculate shot noise using a cubic nuisance function
                        P_SN = self.nuis_cubic(self.z_21, a_sn, b_sn, c_sn, d_sn)
                    else:  # If shot noise is not fitted, extract it from parameters
                        P_SN = []
                        for i in range(len(self.zs_21)):
                            P_SN.append(params_values[f'shot_noise_{i+1}'])
                        P_SN = np.array(P_SN)

                elif self.shot_noise_from_input:  # If shot noise is provided as input
                    P_SN = []
                    for i in range(len(self.zs_21)):
                        P_SN.append(params_values[f'shot_noise_input_{i+1}'])
                    P_SN = np.array(P_SN)  
                else:  # Default case for shot noise
                    P_SN = self.SN_21(self.z_21, self.provider.get_param("H0"), self.provider.get_Hubble(z = self.zs_21, units="km/s/Mpc"))    
            else:
                P_SN = np.zeros_like(self.zs_21)  # No shot noise

            # Handle nuisances for 21 cm observables
            if self.nuisances_21:
                if self.nuisances_fitted:  # If nuisances are fitted
                    if self.nuis_quadratic_fit:  # Check if the nuisance is a quadratic fit
                        # Extract nuisance parameters for quadratic fit
                        a_21_1 = params_values['a_21_1']
                        a_21_2 = params_values['a_21_2']
                        b_21_1 = params_values['b_21_1']
                        b_21_2 = params_values['b_21_2']
                        c_21_1 = params_values['c_21_1']
                        c_21_2 = params_values['c_21_2']
                        # Calculate nuisances using the quadratic function
                        nuis_1 = np.array(self.nuis_quadratic(self.z_21, a_21_1, b_21_1, c_21_1))
                        nuis_2 = np.array(self.nuis_quadratic(self.z_21, a_21_2, b_21_2, c_21_2))        
                    else:  # If nuisance is cubic fit
                        a_21_1 = params_values['a_21_1']
                        a_21_2 = params_values['a_21_2']
                        b_21_1 = params_values['b_21_1']
                        b_21_2 = params_values['b_21_2']
                        c_21_1 = params_values['c_21_1']
                        c_21_2 = params_values['c_21_2']
                        d_21_1 = params_values['d_21_1']
                        d_21_2 = params_values['d_21_2']
                        # Calculate nuisances using the cubic function
                        nuis_1 = np.array(self.nuis_cubic(self.z_21, a_21_1, b_21_1, c_21_1, d_21_1))
                        nuis_2 = np.array(self.nuis_cubic(self.z_21, a_21_2, b_21_2, c_21_2, d_21_2))
                else:  # If nuisances are not fitted, extract from parameters
                    nuis_1 = []
                    nuis_2 = []
                    for i in range(len(self.zs_21)):
                        nuis_1.append(params_values[f'Tbsigma8_{i+1}'])
                        nuis_2.append(params_values[f'Tfsigma8_{i+1}'])
                    nuis_1 = np.array(nuis_1)
                    nuis_2 = np.array(nuis_2)
                
                # Loop over each observable for 21 cm
                for observable in self.observables_21:
                    # Get the theoretical prediction for the observable with nuisances
                    theory_aux = self.get_theory(observable, self.z_21, nuisances_21=self.nuisances_21, nuis_1=nuis_1, nuis_2=nuis_2, P_SN=P_SN)
                    # Get the corresponding data for the observable
                    data_aux = self.get_data(observable)
                    # Store theory and data in auxiliary lists
                    theory_21_aux.append(theory_aux)
                    data_21_aux.append(data_aux)

                # Combine data and theory across all observables
                self.data_21 = [np.hstack([data_21_aux[j][i] for j in range(len(self.observables_21))]) for i in range(self.number_bins_21)]
                self.theory_21 = [np.hstack([theory_21_aux[j][i] for j in range(len(self.observables_21))]) for i in range(self.number_bins_21)]
                # Compute the difference between theory and data
                diff_21 = [t - d for t, d in zip(self.theory_21, self.data_21)]
                # Calculate the result for the 21 cm data
                for i in range(self.number_bins_21):
                    d = np.array(diff_21[i])
                    cov = np.array(self.inv_cov_21[i])
                    aux = np.dot(cov, d)
                    result_21 += np.dot(d.T, aux)

            else:  # If no nuisances are used
                for observable in self.observables_21:
                    # Get the theoretical prediction for the observable without nuisances
                    theory_aux = self.get_theory(observable, self.z_21, nuisances_21=self.nuisances_21, P_SN=P_SN)
                    # Get the corresponding data for the observable
                    data_aux = self.get_data(observable)
                    # Store theory and data in auxiliary lists
                    theory_21_aux.append(theory_aux)
                    data_21_aux.append(data_aux)

                # Combine data and theory across all observables
                self.data_21 = [np.hstack([data_21_aux[j][i] for j in range(len(self.observables_21))]) for i in range(self.number_bins_21)]
                self.theory_21 = [np.hstack([theory_21_aux[j][i] for j in range(len(self.observables_21))]) for i in range(self.number_bins_21)]
                # Compute the difference between theory and data
                diff_21 = [t - d for t, d in zip(self.theory_21, self.data_21)]            
                # Calculate the result for the 21 cm data
                for i in range(self.number_bins_21):
                    d = np.array(diff_21[i])
                    cov = np.array(self.inv_cov_21[i])
                    aux = np.dot(cov, d)
                    result_21 += np.dot(d.T, aux)

        # Check if the object has cross observable data and that it is defined
        if hasattr(self, 'observables_cross') and self.observables_cross:
            
            # Initialize lists to store theory and data for cross observables
            theory_cross_aux = []
            data_cross_aux = []
            self.data_cross = []
            self.theory_cross = []
            
            # If there are nuisances for the cross observables
            if self.nuisances_cross:
                # Check if nuisances are fitted
                if self.nuisances_fitted:
                    # Determine the type of nuisance model to apply
                    if self.nuis_quadratic_fit:
                        # Extract quadratic nuisance parameters from params_values
                        a_cross_1 = params_values['a_cross_1']
                        a_cross_2 = params_values['a_cross_2']
                        a_cross_3 = params_values['a_cross_3']
                        b_cross_1 = params_values['b_cross_1']
                        b_cross_2 = params_values['b_cross_2']
                        b_cross_3 = params_values['b_cross_3']
                        c_cross_1 = params_values['c_cross_1']
                        c_cross_2 = params_values['c_cross_2']
                        c_cross_3 = params_values['c_cross_3']
                        
                        # Calculate nuisances using the quadratic model
                        nuis_1 = np.array(self.nuis_quadratic(self.z_cross, a_cross_1, b_cross_1, c_cross_1))
                        nuis_2 = np.array(self.nuis_quadratic(self.z_cross, a_cross_2, b_cross_2, c_cross_2))
                        nuis_3 = np.array(self.nuis_quadratic(self.z_cross, a_cross_3, b_cross_3, c_cross_3))
                    else:
                        # Extract cubic nuisance parameters from params_values
                        a_cross_1 = params_values['a_cross_1']
                        a_cross_2 = params_values['a_cross_2']
                        a_cross_3 = params_values['a_cross_3']
                        b_cross_1 = params_values['b_cross_1']
                        b_cross_2 = params_values['b_cross_2']
                        b_cross_3 = params_values['b_cross_3']
                        c_cross_1 = params_values['c_cross_1']
                        c_cross_2 = params_values['c_cross_2']
                        c_cross_3 = params_values['c_cross_3']
                        d_cross_1 = params_values['d_cross_1']
                        d_cross_2 = params_values['d_cross_2']
                        d_cross_3 = params_values['d_cross_3']
                        
                        # Calculate nuisances using the cubic model
                        nuis_1 = np.array(self.nuis_cubic(self.z_cross, a_cross_1, b_cross_1, c_cross_1, d_cross_1))
                        nuis_2 = np.array(self.nuis_cubic(self.z_cross, a_cross_2, b_cross_2, c_cross_2, d_cross_2))
                        nuis_3 = np.array(self.nuis_cubic(self.z_cross, a_cross_3, b_cross_3, c_cross_3, d_cross_3))
                else:
                    # If nuisances are not fitted, retrieve them from params_values
                    nuis_1 = []
                    nuis_2 = []
                    nuis_3 = []
                    for i in range(len(self.zs_cross)):
                        nuis_1.append(params_values[f'rTbsigma8_{i+1}'])
                        nuis_2.append(params_values[f'rTfsigma8_{i+1}'])
                        nuis_3.append(params_values[f'rTbgsigma8_{i+1}'])
                    nuis_1 = np.array(nuis_1)
                    nuis_2 = np.array(nuis_2)
                    nuis_3 = np.array(nuis_3)
                    
                # Iterate through each observable for cross correlation
                for observable in self.observables_cross:
                    # Get theoretical predictions for the observable with nuisance parameters
                    theory_aux = self.get_theory(observable, self.z_cross, nuisances_cross=self.nuisances_cross, nuis_1=nuis_1, nuis_2=nuis_2, nuis_3=nuis_3)
                    # Get the corresponding data for the observable
                    data_aux = self.get_data(observable)
                    # Append the results to auxiliary lists
                    theory_cross_aux.append(theory_aux)
                    data_cross_aux.append(data_aux)

                # Combine data and theory from different observables into single arrays
                self.data_cross = [np.hstack([data_cross_aux[j][i] for j in range(len(self.observables_cross))]) for i in range(self.number_bins_cross)]
                self.theory_cross = [np.hstack([theory_cross_aux[j][i] for j in range(len(self.observables_cross))]) for i in range(self.number_bins_cross)]                        
                
                # Calculate the difference between theory and data
                diff_cross = [t - d for t, d in zip(self.theory_cross, self.data_cross)]                    
                for i in range(self.number_bins_cross):
                    d = np.array(diff_cross[i])
                    cov = np.array(self.inv_cov_cross[i])  # Inverse covariance matrix
                    aux = np.dot(cov, d)  # Apply inverse covariance to the difference
                    result_cross += np.dot(d.T, aux)  # Update the result with the dot product

            else:
                # If no nuisances are specified, determine galaxy bias
                if self.galaxy_bias_from_input:
                    b_g = []
                    for i in range(len(self.zs_cross)):
                        b_g.append(params_values[f'galaxy_bias_{i+1}'])
                    b_g = np.array(b_g)  # Create an array for galaxy bias
                else:
                    # Calculate galaxy bias from model
                    b_g = np.array(self.b_g(self.z_cross))
                
                # Iterate through each observable for cross correlation without nuisances
                for observable in self.observables_cross:
                    # Get theoretical predictions for the observable without nuisance parameters
                    theory_aux = self.get_theory(observable, self.z_cross, nuisances_cross=self.nuisances_cross, b_g=b_g)
                    data_aux = self.get_data(observable)  # Get the corresponding data
                    theory_cross_aux.append(theory_aux)
                    data_cross_aux.append(data_aux)

                # Combine data and theory from different observables into single arrays
                self.data_cross = [np.hstack([data_cross_aux[j][i] for j in range(len(self.observables_cross))]) for i in range(self.number_bins_cross)]
                self.theory_cross = [np.hstack([theory_cross_aux[j][i] for j in range(len(self.observables_cross))]) for i in range(self.number_bins_cross)]                       
                
                # Calculate the difference between theory and data
                diff_cross = [t - d for t, d in zip(self.theory_cross, self.data_cross)]                    
                for i in range(self.number_bins_cross):
                    d = np.array(diff_cross[i])
                    cov = np.array(self.inv_cov_cross[i])  # Inverse covariance matrix
                    aux = np.dot(cov, d)  # Apply inverse covariance to the difference
                    result_cross += np.dot(d.T, aux)  # Update the result with the dot product

        # Calculate the final result as the sum of 21cm and cross results
        final_result = result_21 + result_cross
        # Return the log-likelihood (negative half of the final result)
        return -0.5 * final_result

