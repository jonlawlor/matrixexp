// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// General is a typical matrix literal
type General struct {
	blas64.General
}

// Dims returns the matrix dimensions.
func (m1 *General) Dims() (r, c int) {
	r, c = m1.Rows, m1.Cols
	return
}

// At returns the value at a given row, column index.
func (m1 *General) At(r, c int) float64 {
	return m1.Data[r*m1.Stride+c]
}

// Set changes the value at a given row, column index.
func (m1 *General) Set(r, c int, v float64) {
	m1.Data[r*m1.Stride+c] = v
}

// Vector returns all of the values in the matrix as a []float64, in row order.
func (m1 *General) Vector() []float64 {
	// TODO(jonlawlor): make use of a pool.
	v := make([]float64, len(m1.Data))
	copy(v, m1.Data)
	return v
}

func (m1 *General) Eval() Matrix {
	return m1
}

func (m1 *General) T() Matrix {
	return &T{m1}
}

func (m1 *General) Add(m2 Matrix) Matrix {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

func (m1 *General) Sub(m2 Matrix) Matrix {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

func (m1 *General) Mul(m2 Matrix) Matrix {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

func (m1 *General) MulElem(m2 Matrix) Matrix {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

func (m1 *General) DivElem(m2 Matrix) Matrix {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
