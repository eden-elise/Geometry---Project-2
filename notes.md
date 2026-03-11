# Understanding matplotlib and TextPath

When matplotlib encodes a font path, it doesn't just dump a flat list of points. It uses drawing commands to discribe pen movements such as
- MOVETO (lifes pen)
- LINETO (draws a straight line)
- CURVETO (draw a curve to here)

Between letters in words matplotlib needs to life the pen and reposition it, this can be done with MOVETO