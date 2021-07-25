import pickle
import datetime
 
class Data():
    """Data object used to store, save, and load circuits and related data.
    
    When an object is saved, the description and filename is also added to a ledger file (specified when 
    initialising the object). 
    
    Arguments
    ---------
    circuit: Circuit, optional
        A circuit object (QiskitCircuit or MatrixCircuit) to store
    result: Result, optional
        The Qiskit Result object returned when executing a circuit 
    time: string, optional
        The datetime string used to create filenames. If not specified or None, creates the string based on the time
        the object is created (default None)
    desc: string, optional
        A description of the data to be stored (default '')
    project: string, optional
        A title for the overall experiment being run; used as a prefix to the date when creating file
        names (default '')
    folder: string, optional
        Path for a folder in which to save the data (default 'data')
    ledger: string, optional
        The path and name of a txt file in which to append the record of the file when saving (default 'data/ledger.txt')
    **kwargs:
        Additional arguments which are stored in an 'other' variable as dictionary entries
    
    Methods
    -------
    add(key, value):
        Add or modify an entry in the self.other attribute
    save(filename = None):
        Save the data object to a pickle file. If no filename is specified, create one using the project name, folder, and time using the 
        stencil: '[folder]/[project]__[time].pickle'
    from_file(filename):
        Class method used to load an instance of Data() from a file
    """
    def __init__(self, circuit = None, result = None, time = None, desc = "", project = "", folder = "data", ledger = "data/ledger.txt", **kwargs):
        self.circuit = circuit
        self.result = result
        self.time = time if time is not None else datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.desc = desc
        self.project = project
        self.folder = folder
        self.ledger = ledger
        self.other = kwargs
    
    def __repr__(self):
        return f"Date: {self.time}; Project: {self.project}; Description: {self.desc}; Other: {self.other}"
    
    def add(self, key, value):
        """Add or modify an entry in the self.other attribute."""
        self.other[key] = value
        
    def save(self, filename = None, write_ledger = True):
        """Save the data object to a pickle file. 
        
        Arguments
        ---------
        filename: string, optional
            The path for the file where the data will be stored. If no filename is specified, create one using the project name, folder, 
            and time using the stencil: '[folder]/[project]__[time].pickle' (default None)
        write_ledger: bool, optional
            Whether to also append an entry to the ledger file specified when creating the object (default True)
        """
        # Create filename if it is not specified
        if filename is None:
            filename = f"{self.folder}{'/' if self.folder != '' else ''}{self.project}__{self.time}.pickle"
        # Append the correct extension if it is not given already
        if not filename[-7:] == ".pickle":
            filename += ".pickle"
        # Create the file and save the data
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f, protocol = pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)
            
        if write_ledger:
            # Write to the ledger
            title = f"{self.project}__{self.time}"
            try:
                with open(self.ledger, "a") as f:
                    f.write(f"{(title + ':'):<55}{self.desc}\n")
            except Exception as ex:
                print("Error during writing ledger:", ex)
    
    def from_file(filename):
        """Class method used to load an instance of Data() from a file.
        
        Arguments
        ---------
        filename: string
            Path to the .pickle file where the object is stored
        """
        if not filename[-7:] == ".pickle":
            filename += ".pickle"
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as ex:
            print("Error during unpickling object (Possibly unsupported):", ex)