// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package rewrite

import (
	"github.com/jonlawlor/matrixexp"
)

// AnyExp represents any matrix expression.  It is intended for use with the
// Template rewriter.  It implements matrix algebra but if it is ever used for
// calculations it will cause a runtime panic.
type AnyExp struct{}

// Dims returns the matrix dimensions.
func (m1 *AnyExp) Dims() (r, c int) {
	return 0, 0
}

// At returns the value at a given row, column index.
func (m1 *AnyExp) At(r, c int) float64 {
	panic("cannot evaluate an AnyExpr")
}

// Eval returns a matrix literal.
func (m1 *AnyExp) Eval() matrixexp.MatrixLiteral {
	panic("cannot evaluate an AnyExpr")
}

// Copy creates a (deep) copy of the Matrix Expression.
func (m1 *AnyExp) Copy() matrixexp.MatrixExp {
	return &AnyExp{}
}

// Err returns the first error encountered while constructing the matrix expression.
func (m1 *AnyExp) Err() error {
	panic("cannot evaluate an AnyExpr")
}

// T transposes a matrix.
func (m1 *AnyExp) T() matrixexp.MatrixExp {
	return &matrixexp.T{m1}
}

// Add two matrices together.
func (m1 *AnyExp) Add(m2 matrixexp.MatrixExp) matrixexp.MatrixExp {
	return &matrixexp.Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *AnyExp) Sub(m2 matrixexp.MatrixExp) matrixexp.MatrixExp {
	return &matrixexp.Sub{
		Left:  m1,
		Right: m2,
	}
}

// Scale performs scalar multiplication.
func (m1 *AnyExp) Scale(c float64) matrixexp.MatrixExp {
	return &matrixexp.Scale{
		C: c,
		M: m1,
	}
}

// Mul performs matrix multiplication.
func (m1 *AnyExp) Mul(m2 matrixexp.MatrixExp) matrixexp.MatrixExp {
	return &matrixexp.Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *AnyExp) MulElem(m2 matrixexp.MatrixExp) matrixexp.MatrixExp {
	return &matrixexp.MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *AnyExp) DivElem(m2 matrixexp.MatrixExp) matrixexp.MatrixExp {
	return &matrixexp.DivElem{
		Left:  m1,
		Right: m2,
	}
}

// Match determines if a matrix expression wildcard matches another matrix
// expression.
func (m1 *AnyExp) Match(m2 matrixexp.MatrixExp) error {
	// AnyExp matches all other expressions.
	return nil
}
