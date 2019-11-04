##  Projections onto the canonical simplex with additional linear inequalities
This repository is a complementary material to our paper ["Projections onto the canonical simplex with additional linear inequalities"](https://arxiv.org/abs/1905.03488), where we consider three such projections coming from the fields of distributionally robust optimization (DRO) and accuracy at the top (AATP).

For a brief presentation, see the [Jupyter notebook](https://github.com/VaclavMacha/ProjectionsExamples/blob/master/examples.ipynb).

# Installation

This package can be installed using pkg REPL as follows
```julia
(v1.2) pkg> add https://github.com/VaclavMacha/Projections.git#develop
```

# Usage

This package provides an interface to solve the three following problems

1. distributionally robust optimization (DRO) with φ -divergence or norm in constraints. All implemented φ -divergences and norms are presented in the following table

| Name                          | Constructor           | Sadda solver | General solver | Philpott solver |
| ---                           | :---:                 | :---:        | :---:          | :---:           |
| *Kullback-Leibler divergence* | `KullbackLeibler()`   | ✔            | Ipopt          | ✘               |
| *Burg entropy*                | `Burg()`              | ✔            | Ipopt          | ✘               |
| *Hellinger distance*          | `Hellinger()`         | ✔            | Ipopt          | ✘               |
| *χ2-distance*                 | `ChiSquare()`         | ✔            | Ipopt          | ✘               |
| *Modified χ2-distance*        | `ModifiedChiSquare()` | ✔            | Ipopt          | ✘               |
| *l-1 norm*                    | `Lone()`              | ✔            | CPLEX          | ✘               |
| *l-2 norm*                    | `Ltwo()`              | ✔            | CPLEX          | ✘               |
| *l-infinity norm*             | `Linf()`              | ✔            | CPLEX          | ✔               |

2. finding projection onto probability simplex (Simplex).

The interface provides 3 solvers: 

1. `Sadda()` - our approach how to solve DRO
2. `General()` - general purpose solvers such as [Ipopt](https://github.com/coin-or/Ipopt) or [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)
3. `Philpott()` - algorithm presented in [[Philpott 2018]](https://link.springer.com/article/10.1007/s10287-018-0314-0) for the DRO with l-2 norm

## DRO




The following example shows how to solve the DRO with Burg distance using `Sadda()` and `General()` solver
```julia
julia> using Projections, Random, LinearAlgebra                                                                                           
                                                                                                                                          
julia> Random.seed!(1234);                                                                                                                
                                                                                                                                       
julia> d  = Projections.Burg();                                                                                                           
                     
julia> q  = rand(10);                                                                                                                     
                     
julia> q ./= sum(q);                                                                                                                      
                                                  
julia> c  = rand(10);                                                                                                                     
                                                                                        
julia> ε  = 0.1;                                                                                                                          
                                                                                                      
julia> model  = Projections.DRO(d, q, c, ε);                                                                                              
                       
julia> p1 = Projections.solve(Sadda(), model);                                                                                            
                                                                                                                                          
julia> p2 = Projections.solve(General(), model);                                                                                          

julia> LinearAlgebra.norm(p1 - p2)                                                                                                        
7.848203730531935e-9
```

## Simplex

The following example shows how to solve the Simplex problem using `Sadda()` and `General()` solver
```julia
julia> using Projections, Random, LinearAlgebra                                                                                      

julia> Random.seed!(1234);                                                                                                           

julia> n  = 10;                                                                                                                      

julia> q  = rand(n);                                                                                                                 

julia> lb = rand(n)./n;                                                                                                              

julia> ub = 1 .+ rand(n);                                                                                                            

julia> model  = Projections.Simplex(q, lb, ub);                                                                                     

julia> p1 = Projections.solve(Sadda(), model);                                                                                       

julia> p2 = Projections.solve(General(), model);                                                                                     

julia> LinearAlgebra.norm(p1 - p2)                                                                                                   
2.9749633079816928e-8
```