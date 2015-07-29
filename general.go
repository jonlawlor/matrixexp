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

// Dims returns the matrix dimensions
func (g *General) Dims() (r, c int) {
	r, c = g.Rows, g.Cols
	return
}
