# Εθνικό και Καποδιστριακό Πανεπιστήμιο Αθηνών
### Τμήμα Πληροφορικής και Τηλεπικοινωνιών
### ΘΠ04 - Παράλληλα Συστήματα
### **Εργασία - Game of Life**

**Μέλη**:
 - Βασίλειος Πουλόπουλος - 1115201600141
 - Κωνσταντίνος Χατζόπουλος - 1115201300202

## Εισαγωγή

Για την υλοποίηση της εργασίας, δημιουργήσαμε αρχεία εισόδου για τις μετρήσεις. Για κάθε τμήμα της εργασίας δημιουργήσαμε ξεχωριστο φάκελο στον οποίο υπάρχει ο κώδικας, οι μετρήσεις, τα αποτελέσματα του profiling, τα input files και κάποια bash scripts που υλοποιήσαμε για να αυτοματοποιήσουμε τις παρακάτω διαδικασίες:
 - μεταγλώτηση
 - εκτέλεση στην argo
 - συλλογή μετρήσεων
 - υπολογισμό speedup   
 - υπολογισμό efficiency.
 
Τα scripts αρχικά μεταγλωτίζουν το πρόγραμμα με τις αντίστοιχες παραμέτρους κάθε φορά (μέγεθος προβλήματος **Ν**, αριθμό **διεργασιών** / **nodes** / **cpus** / **threads** / **gpus**) και στη συνέχεια για να το τρέξουν το τοποθετούν στην ουρά με την εντολή **qsub**.

Για όσο βρίσκεται στην ουρά και εκτελείται ή περιμένει να εκτελεστεί, το script ελέγχει σταδιακά με την εντολή qstat αν ολοκληρώθηκε το **job** και αμέσως μετά, με την εντολή append του bash (**>>**) δημιουργεί (αν δεν υπάρχουν ήδη) αρχεία που περιέχουν μόνο τους χρόνους με όνομα **times.txt**. Τέλος για τα script που αφορούν το **MPI** ή **MPI + OpenMp**, το script διαβάζει τα αντίστοιχα αρχεία χρόνων και δημιουργεί τα **speedup.txt** και **efficiency.txt**.

Επίσης, υλοποιήσαμε script που δημιουργεί **επίπεδα αρχεία εισόδου** (όλος ο δισδιάστατος πίνακας χαρακτήρων σε μια γραμμή) τα οποία είναι ίδια με τα **παραγόμενα αρχεία κάθε βήματος**. Επιπλέον για να ελέγξουμε αν κάθε βήμα έχει υπολογιστεί σωστά υλοποιήσαμε script που μετατρέπει τα επιπεδα αρχεία **εισόδου** / **εξόδου** σε δισδιάστατη αναπαράσταση μαύρων ή άσπρων κουτιών (μαύρο κουτί αν η τιμή είναι 0 και άσπρο αν είναι 1).

Στο **MPI** και **MPI + OpenMp** χρησιμοποιούμε command line arguments για να πάρουμε τα εξής:
-   path για το input file
-   path για το output file
-   αριθμό γραμμών
-   αριθμό στηλών

## MPI

Η επιλογή των κόμβων και των διεργασιών έγινε με το σκεπτικό να γεμίζει ο κάθε κόμβος πριν χρησιμοποιήσουμε κάποιο καινούργιο. Αυτό το κάνουμε για να έχουμε όσο το δυνατό λιγότερη επικοινωνία ανάμεσα στους κόμβους ώστε να αποφεύγονται άσκοπες καθυστερήσεις. Δοκιμάζοντας και τις 2 προσεγγίσεις (**πληρότητα κόμβων**, **διασπορά διεργασιών σε κόμβους**) καταλήξαμε στην πρώτη διότι παρατηρήσαμε ότι ήταν πιο γρήγορη, πράγμα το οποίο είναι λογικό αφού στην αρχιτεκτονική κατανεμημένης μνήμης ο κάθε κόμβος δεν έχει πρόσβαση στη μνήμη άλλου κόμβου. Αυτό έχει ως αποτέλεσμα όσο τους αυξάνουμε να αυξάνεται και η επικοινωνία μεταξύ των κόμβων μέσω μηνυμάτων και συνεπως να αυξάνεται η καθυστέρηση.

### Σχεδιασμός

Ο δισδιάστατος πίνακας εισόδου θεωρείται **περιοδικός**, π.χ. αν είμαστε στην τελευταία γραμμή του δισδιάστατου και θέλουμε να πάμε στην ακριβώς από κάτω του, τότε αυτή θα είναι η πρώτη γραμμή κ.ο.κ.

Για την καλύτερη επίτευξη επικοινωνίας μεταξύ των κόμβων, δημιουργούμε μια τοπολογία όπου περιέχει ένα δισδιάστατο πλέγμα από blocks / διεργασίες διαστάσεων **sqrt( # of processes ) * sqrt( # of processes )**. Με αυτόν τον τρόπο, κάθε διεργασία είναι στοιχισμένη σε κάποια θέση στο δισδιάστατο πλέγμα και επεξεργάζεται ένα μόνο block / υποπίνακα του συνολικού προβλήματος.

### Σχεδιασμός MPI κώδικα
Για να μπορέσει να τρέξει το MPI πρόγραμμα, πρέπει να γίνει αρχικοποίηση. Για να γινει αυτό καλούμε τη συνάρτηση **MPI_Init()**. Αμέσως μετά δημιουργούμε την τοπολογία μέσω της συνάρτησης **setupGrid()** όπου αρχικοποιεί τη δομή **gridInfo** με τα δεδομένα της αντίστοιχης διεργασίας.

**struct gridInfo:**
```c
typedef struct gridInfo {  
  MPI_Comm gridComm;		// communicator for entire grid  
  Neighbors neighbors;		// neighbor processes  
  int processes;			// total number of processes  
  int gridRank;				// rank of current process in gridComm  
  int gridDims[2];			// grid dimensions  
  int gridCoords[2];		// grid coordinates  
  int blockDims[2]; 		// block dimensions  
  int localBlockDims[2];	// local block dimensions  
  int stepGlobalChanges;  
  int stepLocalChanges;  
} GridInfo;
```
Τα δεδομένα που αρχικοποιεί η συνάρτηση **setupGrid()**  σε κάθε διεργασία είναι:
 - ο communicator που ανήκει στο πλέγμα,   οι γειτονικές διεργασίες,   
 - ο συνολικός αριθμός διεργασιών
 - η τάξη της διεργασίας στο πλέγμα,   
 - τις διαστάσεις του πλέγματος
 - τις συντεταγμένες της στο πλέγμα  καθώς και
 - τις διαστάσεις του συνολικού πίνακα αλλά και του
 - τοπικού (δικού της) πίνακα.

Για να δημιουργηθεί η τοπολογία καλούμε την συνάρτηση **MPI_Cart_create()** και για να λάβουμε τις συντεταγμένες κάθε διεργασίας την **MPI_Cart_coords()**.

Για τον υπολογισμό κάθε βήματος χρειάζεται να ξέρουμε την προηγούμενη κατάσταση του παιχνιδιού για  να υπολογίσουμε την τρέχουσα. Για το λόγο αυτό δεσμέυουμε δυναμικά με χρήση της συνάρτησης **malloc()** 2 δισδιάστατους πίνακες χαρακτήρων (**old**, **current**) διαστάσεων `grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2` δηλαδή τις διαστάσεις που αντιστοιχούν στον κάθε υποπίνακα του συνολικού προβλήματος **αυξημένο κατα 2*2 διαστάσεις** για να συμπεριληφθούν και οι γειτονικές τιμές.

#### MPI Datatypes

Επειδή υπήρχε η ανάγκη να στέλνουμε και να λαμβάνουμε γραμμές και στήλες από τους τοπικούς αυτούς πίνακες σε γειτονικές διεργασίες, δημιουργήσαμε **datatypes** με τις κατάλληλες συναρτήσεις (**MPI_Type_vector()**, **MPI_Type_commit()**) γιατί χρειαζόμασταν ένα τρόπο με τον οποίο ή κάθε διεργασία θα αποθηκεύει στον δικό της πίνακα τις τιμές των πινάκων των γειτονικών block για να μπορει να υπολογίσει τις ακριανές τιμές του πίνακά της.

Το MPI θεωρεί πως για τα datatypes τα δεδομένα του buffer αποστολής / λήψης πρέπει να είναι σε συνεχόμενες θέσεις μνήμης, και τα offset για γραμμές και στήλες να είναι σταθερα, οπότε χρειαζόταν οι δισδιάστατοι πίνακες **old** και **current** να είναι σε συνεχόμενες θέσεις μνήμης. Γι αυτό το λόγο υλοποιήσαμε την συνάρτηση **allocate2DArray()** με τέτοιο τρόπο ώστε να δημιουργεί ένα δισδιάστατο πίνακα σε συνεχόμενες θέσεις μνήμης.

#### Είσοδος δεδομένων αρχικής κατάστασης παιχνιδιού

Αν υπάρχει αρχείο εισόδου χρησιμοποιούμε ένα **subarray** datatype που δημιουργήσαμε για την ανάγνωση των σωστών περιοχών/τμημάτων του αρχείου εισόδου από το αντίστοιχο process μέσω της **MPI_File_set_view()** και το διαβάζουμε μέσω της **MPI_File_read()**. Στην περίπτωση που δεν υπάρχει αρχείο εισόδου, δημιουργείται ένας τυχαίος πίνακας με ολόκληρη την αρχική κατάσταση του παιχνιδιού μέσω της συνάρτησης **initialize_block()** και με τη βοήθεια της συνάρτησης int **scatter2DArray()** (*κώδικας από το βιβλίο MPI ΘΕΩΡΙΑ ΚΑΙ ΕΦΑΡΜΟΓΕΣ*), ο πίνακας διασπείρεται σε όλες τις διεργασίες του δίσδιάστατου πλέγματος. Με αυτόν τον τρόπο η κάθε διεργασία έχει στον τοπικό της υποπίνακα '**old**' το αντίστοιχο υποπρόβλημα που καλείται να υπολογίσει.

#### Επικοινωνία μεταξύ των διεργασιών του πλέγματος (Κεντρική επανάλληψη)

Για την επίτευξη της αποστολής των 8 μηνυμάτων (ένα για κάθε γείτονα) στις γειτονικές διεργασίες καθώς και τη λήψη άλλων 8, χωρίς την επιβάρυνση της κεντρικής επανάληψης με επιπλέον υπολογισμούς (offsets, κτλ), απαιτείται η αρχικοποίηση τους μέσω της συνάρτησης **MPI_Send_init()** ώστε να δημιουργηθούν 16 requests τα οποία θα είναι έτοιμα προς εκκίνηση όταν αυτό χρειαστεί με τη χρήση της συνάρτησης **MPI_Startall()**.  Στη συνέχεια γίνονται οι υπολογισμοί των εσωτερικών κυττάρων και κατόπιν, καλείται η **MPI_Waitall()** ωστε πριν υπολογιστούν και τα εξωτερικά κύτταρα να έχουν σταλθεί όλα τα γειτονικά σε κάθε block για να γίνουν σωστά οι υπολογισμοί. 

#### Υπολογισμοί

Η συνάρτηση **calculate()** που έχουμε υλοποιήσει για τους υπολογισμούς είναι **inline** για να μην υπάρχουν καθυστερήσεις, επιπλέον, η συνάρτηση αυτή δέχεται ώς όρισμα ένα δείκτη στην ακέραια μεταβλητή **changes** όπου αν υπάρξει αλλαγή στο συγκεκριμένο κύτταρο που εξετάζεται εκείνη τη στιγμή, η μεταβλητή αυτή αυξάνεται κατά 1. 

#### Swap

Στο τέλος της επανάληψης, το αποτέλεσμα που υπολογίστηκε από τον πίνακα '**old'**, θα αποθηκευτεί στον πινακα '**current**'. Έτσι, στην επόμενη επανάληψη θα πρέπει η παλιά κατάσταση του παιχνιδιού να αντικατασταθεί με την current του προηγούμενου βήματος. Για το λόγο αυτό, πρέπει να γίνει **swap** του old με τον current.

### Μετρήσεις χρόνων εκτέλεσης (Χωρίς reduce)

| |1 Process| 4 Processes| 16 Processes| 64 Processes| 
|--|--|--|--|--| 
| **320x320**     | 0.745   |  **0.216**   | 0.202   | 0.292 |
| **640x640**     | 3.354   |  0.770   | **0.313**   | 0.310 |
| **1280x1280**   | 13.268  |  3.383   | 0.851   | **0.391** |
| **2560x2560**   | 53.610  |  13.459  | 3.466   | **0.917** |
| **5120x5120**   | 213.743 |  53.943  | 18.438  | **8.652** |
| **10240x10240** | -       |  215.087 | 54.659  | **13.750** |
| **20480x20480** | -       |  -       | 217.871 | **57.988** |
| **40960x40960** | -       |  -       | -       | **218.213**|

Παρατηρούμε ότι όσο αυξάνεται το μέγεθος του προβλήματος, μεγαλώνοντας τον αριθμό των διεργασιών ότι στα πράσινα υπάρχει καλύτερη κλιμάκωση, συνεχίζοντας όμως να μεγαλώνουμε τον αριθμό, βλέπουμε ότι δεν συμφέρει (κίτρινα) διότι γίνεται σπατάλη πόρων για την επίτευξη του ίδιου περίπου χρόνου με τα πράσινα.

Επιπλέον, για μεγέθη μεγαλύτερα από 5120x5120 δεν έφτασε το όριο των 10 λεπτών και σταμάτησε η εκτέλεση.