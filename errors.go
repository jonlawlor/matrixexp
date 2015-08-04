// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import "fmt"

// Errors related to literals.

// ErrInvalidRows happens when a literal has a negative number of rows.
type ErrInvalidRows int

func (e ErrInvalidRows) Error() string {
	return fmt.Sprintf("invalid rows: %d", e)
}

// ErrInvalidCols happens when a literal has a negative number of columns.
type ErrInvalidCols int

func (e ErrInvalidCols) Error() string {
	return fmt.Sprintf("invalid cols: %d", e)
}

// ErrInvalidStride happens when a literal has a nonpositive stride.
type ErrInvalidStride int

func (e ErrInvalidStride) Error() string {
	return fmt.Sprintf("invalid stride: %d", e)
}

// ErrStrideLessThanCols happens when a literal has stride less than columns.
type ErrStrideLessThanCols struct {
	Stride, Cols int
}

func (e ErrStrideLessThanCols) Error() string {
	return fmt.Sprintf("invalid stride: %d < cols %d", e.Stride, e.Cols)
}

// ErrInvalidDataLen happens when the slice of data is too small to fit all of
// the values that can be addressed in the matrix.
type ErrInvalidDataLen struct {
	GotLen, WantLen int
}

func (e ErrInvalidDataLen) Error() string {
	return fmt.Sprintf("invalid data len: %d < addressable %d", e.GotLen, e.WantLen)
}

// ErrDimMismatch happens when two matrices cannot be operated on due to
// differnt sizes.  For example, you can't perform element-wise multiplication
// between a 2x2 matrix and a 3x3 matrix.
type ErrDimMismatch struct {
	R1, C1 int
	R2, C2 int
}

func (e ErrDimMismatch) Error() string {
	return fmt.Sprintf("dimension mismatch: (%d, %d) vs (%d, %d)", e.R1, e.C1, e.R2, e.C2)
}

// ErrInnerDimMismatch happens when you try to use matrix multiplication on two
// matrices that have different inner dimensions.
type ErrInnerDimMismatch struct {
	C, R int
}

func (e ErrInnerDimMismatch) Error() string {
	return fmt.Sprintf("inner dimension mismatch: %d vs %d", e.C, e.R)
}
