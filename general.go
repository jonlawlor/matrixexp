// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// General is a typical matrix literal.
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

// Eval returns a matrix literal.
func (m1 *General) Eval() MatrixLiteral {
	return m1
}

// T transposes a matrix.
func (m1 *General) T() MatrixExpr {
	return &T{m1}
}

// Add two matrices together.
func (m1 *General) Add(m2 MatrixExpr) MatrixExpr {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *General) Sub(m2 MatrixExpr) MatrixExpr {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Mul performs matrix multiplication.
func (m1 *General) Mul(m2 MatrixExpr) MatrixExpr {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *General) MulElem(m2 MatrixExpr) MatrixExpr {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *General) DivElem(m2 MatrixExpr) MatrixExpr {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}

// Vector returns all of the values in the matrix as a []float64, in row order.
func (m1 *General) AsVector() []float64 {
	// TODO(jonlawlor): make use of a pool.
	v := make([]float64, len(m1.Data))
	copy(v, m1.Data)
	return v
}

// Vector returns all of the values in the matrix as a []float64, in row order.
func (m1 *General) AsGeneral() blas64.General {
	return m1.General
}
