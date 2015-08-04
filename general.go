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

// Copy creates a (deep) copy of the Matrix Expression.
func (m1 *General) Copy() MatrixExp {
	v := make([]float64, len(m1.Data))
	copy(v, m1.Data)
	return &General{
		blas64.General{
			Rows:   m1.Rows,
			Cols:   m1.Cols,
			Stride: m1.Stride,
			Data:   v,
		},
	}
}

// Err returns the first error encountered while constructing the matrix expression.
func (m1 *General) Err() error {
	if m1.Rows < 0 {
		return ErrInvalidRows(m1.Rows)
	}
	if m1.Cols < 0 {
		return ErrInvalidCols(m1.Cols)
	}
	if m1.Stride < 1 {
		return ErrInvalidStride(m1.Stride)
	}
	if m1.Stride < m1.Cols {
		return ErrStrideLessThanCols{m1.Stride, m1.Cols}
	}
	if maxLen := (m1.Rows-1)*m1.Stride + m1.Cols; maxLen > len(m1.Data) {
		return ErrInvalidDataLen{len(m1.Data), maxLen}
	}
	return nil
}

// T transposes a matrix.
func (m1 *General) T() MatrixExp {
	return &T{m1}
}

// Add two matrices together.
func (m1 *General) Add(m2 MatrixExp) MatrixExp {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *General) Sub(m2 MatrixExp) MatrixExp {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Scale performs scalar multiplication.
func (m1 *General) Scale(c float64) MatrixExp {
	C := new(float64)
	*C = c
	return &Scale{
		C: C,
		M: m1,
	}
}

// Mul performs matrix multiplication.
func (m1 *General) Mul(m2 MatrixExp) MatrixExp {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *General) MulElem(m2 MatrixExp) MatrixExp {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *General) DivElem(m2 MatrixExp) MatrixExp {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}

// AsVector returns a copy of the values in the matrix as a []float64, in row order.
func (m1 *General) AsVector() []float64 {
	// TODO(jonlawlor): make use of a pool.
	v := make([]float64, m1.Rows*m1.Cols)
	for i := 0; i < m1.Rows; i++ {
		copy(v[i*m1.Cols:(i+1)*m1.Cols], m1.Data[i*m1.Stride:i*m1.Stride+m1.Cols])
	}
	copy(v, m1.Data)
	return v
}

// AsGeneral returns the matrix as a blas64.General (not a copy!)
func (m1 *General) AsGeneral() blas64.General {
	return m1.General
}
