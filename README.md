# Computer Vision projects – Ermis Efthymiou Mouzouras (AM 1100548)

Κάθε φάκελος: `assignment/` (εκφώνηση), `report/` (η αναφορά μου), `code/` (κώδικας + δεδομένα).

| # | Project | Γλώσσα / εργαλεία |
|---|---|---|
| 01 | Gaussian/Laplacian πυραμίδες, image blending | Python, NumPy |
| 02 | Γεωμετρικοί μετασχηματισμοί (affine, projective) | MATLAB |
| 03 | PCA και autoencoder (MNIST) | Python, PyTorch, scikit-learn |
| 04 | Detection: Faster R-CNN vs SSD300 στο COCO | Python, PyTorch/torchvision |
| 05 | Επεξεργασία εικόνας: Fourier, φίλτρα, Wiener, Canny/Hough | Python, OpenCV, SciPy |
| 06 | Οπτική ροή Lucas–Kanade, γωνίες, SIFT/SURF/FAST, RANSAC, video stabilization | MATLAB |
| 07 | Tracking με particle filter και Kalman | MATLAB |
| 08 | Αναγνώριση χειρονομιών: motion energy + k-NN, RNN, CNN→RNN | Python, OpenCV, PyTorch |
| 09 | K-SVD: denoising και inpainting εικόνων | MATLAB |
| 10 | Αναγνώριση προσώπων: PCA/LDA (Eigenfaces/Fisherfaces) | Python, OpenCV, scikit-learn |
| 11 | Ταξινόμηση προσώπων και φύλου (Bayes) | MATLAB |

## Σημειώσεις
- **04:** οι εικόνες COCO `val2017` (~800 MB) δεν περιλαμβάνονται. Για νέο inference κατέβασέ τις από https://cocodataset.org και βάλ' τις στο `code/val2017/`. Τα αποτελέσματα (`*.pth`) και τα annotations υπάρχουν.
- **05:** το notebook διαβάζει `./images/...` (πεζά). Σε macOS δουλεύει, σε Linux χρειάζεται μετονομασία του `Images` σε `images`.
- **04, 03:** τα notebooks γράφτηκαν για Google Colab/Jupyter με PyTorch.
