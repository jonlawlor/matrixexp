// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

// Not yet implemented.  I'm hoping to use getrf -> getri from gonum/lapack, but
// getri is not yet implemented.  There will be some other speedbumps along the
// way (specifically, the At function and also making sane test cases) and Inv
// isn't all that useful, so I'm happy to leave it for now.
