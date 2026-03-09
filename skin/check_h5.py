import h5py

for fname in ["Result/results.h5", "Result/results_.h5"]:
    print(f"\n{'='*50}\n{fname}")
    with h5py.File(fname, "r") as f:
        def show(name, obj):
            if hasattr(obj, 'shape'):
                print(f"  {name}: {obj.shape}")
        f.visititems(show)