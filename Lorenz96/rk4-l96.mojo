# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

from math import iota
from sys import num_physical_cores, simdwidthof

import benchmark
from algorithm import parallelize, vectorize
from complex import ComplexFloat64, ComplexSIMD
from memory import UnsafePointer, memset_zero
from tensor import Tensor

alias int_type = DType.int32

# Set up the parameters

alias N = 8.0
alias F = 8.0

alias t0 = 0 
alias tn = 5000
alias h = 0.005

alias dXdt = Tensor[DType.float32]
alias total_steps = (tn - t0) / h

alias t = Tensor[DType.float32](0,0)

alias float_type = DType.float32
#struct Matrix[type: DType, rows: Int, cols: Int]:
#    var data: UnsafePointer[Scalar[type]]
#
#    fn __init__(out self):
#        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
#
#    fn store[nelts: Int](self, row: Int, col: Int, val: SIMD[type, nelts]):
#        self.data.store(row * cols + col, val)

fn lorenz96(t: Tensor[DType.float32]) -> Tensor[DType.float32]:
    # Convert dimensions to Int and create tensor
    var dXdt = Tensor[DType.float32](8)  

#   dXdt = Tensor[DType.float32](1000000, 8)  
    for i in range (N):
        dXdt[i] =  F       
    return dXdt

fn main() raises:
    
    print("PRINT THE PARAMETERS")
    print("the number of dimensions is: ", N)
    print("the forcing is: ", F)
    print("the time interval is from ", t0," to ", tn," with a step of ", h)
    print("Givin us a total of  ",total_steps ," steps")
    
    # Define N as Int
    print(t)
    Lorenz = lorenz96(t)
    print(Lorenz)

