wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.2-linux-x86_64.tar.gz

tar -xvf julia-1.5.2-linux-x86_64.tar.gz
./julia-1.5.2/bin/julia --project utrain.jl $PORT
