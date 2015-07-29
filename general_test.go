// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
	"testing"
)

func TestGeneral(t *testing.T) {
	g := General{blas64.General{
		Rows:   3,
		Cols:   3,
		Stride: 3,
		Data: []float64{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		},
	}}
	r, c := g.Dims()
	if r != 3 || c != 3 {
		t.Errorf("Expected Dims() to return (%d, %d), got (%d, %d)", 3, 3, r, c)
	}
}
