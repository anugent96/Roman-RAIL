General Notes:

RAIL dictionary:
dict(
    zmin=0.0,
    zmax=6.0,
    nzbins=601,
    trainfrac=0.75,
    bumpmin=0.02,
    bumpmax=0.35,
    nbump=20,
    sharpmin=0.7,
    sharpmax=2.1,
    nsharp=15,
    max_basis=35,
    basis_system="cosine",
    regression_params={"max_depth": 8, "objective": "reg:squarederror"},
    bands=bands,
    ref_band='roman_R062',
    err_bands=errbands, 
    mag_limits=maglims,
    nondetect_val = 99,
    include_mag_errors=True
)

We want to include magnitude errors in the photometric redshift estimation: this should give more realistic photo-z PDFs. Don't change any of the bump, sharp, or basis parameters: these control the shape of the PDF. The reference band directly control what band is used to determine colors in the photo-z code. I will list notes below for what I think the best reference band is for each photo-z code.
