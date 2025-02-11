# export NERFSTUDIO_METHOD_CONFIGS="splatfacto_scannetpp=splatfacto_scannetpp.splatfacto_scannetpp_config:splatfacto_scannetpp_method"
ns-train instantngp_scannetpp \
        --experiment-name instantngp_scannetpp_undistorted \
        scannetpp-data \
        --data ../data/scannetpp/data/cc5237fd77 \
        --images-dir dslr/undistorted_images \
        --masks-dir dslr/undistorted_anon_masks \
        --transforms-path dslr/nerfstudio/transforms_undistorted.json \
