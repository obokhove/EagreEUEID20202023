# state file generated using paraview version 5.13.3
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2042, 856]
renderView1.InteractionMode = '2D'
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.CenterOfRotation = [5.49, 0.5, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [6.833210374221071, -0.21768548202484753, 21.376477579317328]
renderView1.CameraFocalPoint = [6.833210374221071, -0.21768548202484753, 0.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 3.2071145415460487
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.PolarGrid = 'Polar Grid Actor'
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(2042, 856)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
sw_beachpvd = PVDReader(registrationName='sw_beach.pvd', FileName='/Users/onnobokhove/amtob/werk/vuurdraak2021/EagreEUEID20202023/coupledwt2025/data/FG_FWF_8April/sw_beach.pvd')
sw_beachpvd.PointArrays = ['beach_sw']

# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=sw_beachpvd)
calculator2.Function = 'jHat*beach_sw'

# create a new 'Warp By Vector'
warpByVector1 = WarpByVector(registrationName='WarpByVector1', Input=calculator2)
warpByVector1.Vectors = ['POINTS', 'Result']

# create a new 'PVD Reader'
sw_wavespvd = PVDReader(registrationName='sw_waves.pvd', FileName='/Users/onnobokhove/amtob/werk/vuurdraak2021/EagreEUEID20202023/coupledwt2025/data/FG_FWF_8April/sw_waves.pvd')
sw_wavespvd.PointArrays = ['u_sw']

# create a new 'PVD Reader'
dw_wavespvd = PVDReader(registrationName='dw_waves.pvd', FileName='/Users/onnobokhove/amtob/werk/vuurdraak2021/EagreEUEID20202023/coupledwt2025/data/FG_FWF_8April/dw_waves.pvd')
dw_wavespvd.PointArrays = ['phi_dw']

# create a new 'Gradient'
gradient1 = Gradient(registrationName='Gradient1', Input=dw_wavespvd)
gradient1.ScalarArray = ['POINTS', 'phi_dw']

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=gradient1)
calculator1.Function = 'Gradient_X'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from warpByVector1
warpByVector1Display = Show(warpByVector1, renderView1, 'UnstructuredGridRepresentation')

# get separate 2D transfer function for 'beach_sw'
separate_warpByVector1Display_beach_swTF2D = GetTransferFunction2D('beach_sw', warpByVector1Display, separate=True)

# get separate color transfer function/color map for 'beach_sw'
separate_warpByVector1Display_beach_swLUT = GetColorTransferFunction('beach_sw', warpByVector1Display, separate=True)
separate_warpByVector1Display_beach_swLUT.TransferFunction2D = separate_warpByVector1Display_beach_swTF2D
separate_warpByVector1Display_beach_swLUT.RGBPoints = [0.800125, 0.231373, 0.298039, 0.752941, 0.9499999999999702, 0.865003, 0.865003, 0.865003, 1.0998749999999404, 0.705882, 0.0156863, 0.14902]
separate_warpByVector1Display_beach_swLUT.ScalarRangeInitialized = 1.0

# get separate opacity transfer function/opacity map for 'beach_sw'
separate_warpByVector1Display_beach_swPWF = GetOpacityTransferFunction('beach_sw', warpByVector1Display, separate=True)
separate_warpByVector1Display_beach_swPWF.Points = [0.800125, 0.0, 0.5, 0.0, 1.0998749999999404, 1.0, 0.5, 0.0]
separate_warpByVector1Display_beach_swPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
warpByVector1Display.Representation = 'Surface'
warpByVector1Display.ColorArrayName = ['POINTS', 'beach_sw']
warpByVector1Display.LookupTable = separate_warpByVector1Display_beach_swLUT
warpByVector1Display.SelectNormalArray = 'None'
warpByVector1Display.SelectTangentArray = 'None'
warpByVector1Display.SelectTCoordArray = 'None'
warpByVector1Display.TextureTransform = 'Transform2'
warpByVector1Display.OSPRayScaleArray = 'beach_sw'
warpByVector1Display.OSPRayScaleFunction = 'Piecewise Function'
warpByVector1Display.Assembly = ''
warpByVector1Display.SelectedBlockSelectors = ['']
warpByVector1Display.SelectOrientationVectors = 'Result'
warpByVector1Display.ScaleFactor = 0.2999999999999403
warpByVector1Display.SelectScaleArray = 'beach_sw'
warpByVector1Display.GlyphType = 'Arrow'
warpByVector1Display.GlyphTableIndexArray = 'beach_sw'
warpByVector1Display.GaussianRadius = 0.014999999999997016
warpByVector1Display.SetScaleArray = ['POINTS', 'beach_sw']
warpByVector1Display.ScaleTransferFunction = 'Piecewise Function'
warpByVector1Display.OpacityArray = ['POINTS', 'beach_sw']
warpByVector1Display.OpacityTransferFunction = 'Piecewise Function'
warpByVector1Display.DataAxesGrid = 'Grid Axes Representation'
warpByVector1Display.PolarAxes = 'Polar Axes Representation'
warpByVector1Display.ScalarOpacityFunction = separate_warpByVector1Display_beach_swPWF
warpByVector1Display.ScalarOpacityUnitDistance = 0.2837165114084961
warpByVector1Display.OpacityArrayName = ['POINTS', 'beach_sw']
warpByVector1Display.SelectInputVectors = ['POINTS', 'Result']
warpByVector1Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
warpByVector1Display.ScaleTransferFunction.Points = [0.800125, 0.0, 0.5, 0.0, 1.0998749999999404, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
warpByVector1Display.OpacityTransferFunction.Points = [0.800125, 0.0, 0.5, 0.0, 1.0998749999999404, 1.0, 0.5, 0.0]

# set separate color map
warpByVector1Display.UseSeparateColorMap = True

# show data from calculator1
calculator1Display = Show(calculator1, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'Result'
resultTF2D = GetTransferFunction2D('Result')
resultTF2D.ScalarRangeInitialized = 1
resultTF2D.Range = [-0.1, 0.20000000000000004, 0.0, 1.0]

# get color transfer function/color map for 'Result'
resultLUT = GetColorTransferFunction('Result')
resultLUT.TransferFunction2D = resultTF2D
resultLUT.RGBPoints = [-0.1, 0.231373, 0.298039, 0.752941, 0.05000000000000002, 0.865003, 0.865003, 0.865003, 0.20000000000000004, 0.705882, 0.0156863, 0.14902]
resultLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Result'
resultPWF = GetOpacityTransferFunction('Result')
resultPWF.Points = [-0.1, 0.0, 0.5, 0.0, 0.20000000000000004, 1.0, 0.5, 0.0]
resultPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'
calculator1Display.ColorArrayName = ['POINTS', 'Result']
calculator1Display.LookupTable = resultLUT
calculator1Display.SelectNormalArray = 'None'
calculator1Display.SelectTangentArray = 'None'
calculator1Display.SelectTCoordArray = 'None'
calculator1Display.TextureTransform = 'Transform2'
calculator1Display.OSPRayScaleArray = 'Result'
calculator1Display.OSPRayScaleFunction = 'Piecewise Function'
calculator1Display.Assembly = ''
calculator1Display.SelectedBlockSelectors = ['']
calculator1Display.SelectOrientationVectors = 'Gradient'
calculator1Display.ScaleFactor = 1.102
calculator1Display.SelectScaleArray = 'Result'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'Result'
calculator1Display.GaussianRadius = 0.055099999999999996
calculator1Display.SetScaleArray = ['POINTS', 'Result']
calculator1Display.ScaleTransferFunction = 'Piecewise Function'
calculator1Display.OpacityArray = ['POINTS', 'Result']
calculator1Display.OpacityTransferFunction = 'Piecewise Function'
calculator1Display.DataAxesGrid = 'Grid Axes Representation'
calculator1Display.PolarAxes = 'Polar Axes Representation'
calculator1Display.ScalarOpacityFunction = resultPWF
calculator1Display.ScalarOpacityUnitDistance = 0.9164838448569989
calculator1Display.OpacityArrayName = ['POINTS', 'Result']
calculator1Display.SelectInputVectors = ['POINTS', 'Gradient']
calculator1Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from sw_wavespvd
sw_wavespvdDisplay = Show(sw_wavespvd, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'u_sw'
u_swTF2D = GetTransferFunction2D('u_sw')
u_swTF2D.ScalarRangeInitialized = 1
u_swTF2D.Range = [-0.1, 0.2, 0.0, 1.0]

# get color transfer function/color map for 'u_sw'
u_swLUT = GetColorTransferFunction('u_sw')
u_swLUT.TransferFunction2D = u_swTF2D
u_swLUT.RGBPoints = [-0.1, 0.231373, 0.298039, 0.752941, 0.05000000000000002, 0.865003, 0.865003, 0.865003, 0.20000000000000004, 0.705882, 0.0156863, 0.14902]
u_swLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'u_sw'
u_swPWF = GetOpacityTransferFunction('u_sw')
u_swPWF.Points = [-0.1, 0.0, 0.5, 0.0, 0.20000000000000004, 1.0, 0.5, 0.0]
u_swPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
sw_wavespvdDisplay.Representation = 'Surface'
sw_wavespvdDisplay.ColorArrayName = ['POINTS', 'u_sw']
sw_wavespvdDisplay.LookupTable = u_swLUT
sw_wavespvdDisplay.SelectNormalArray = 'None'
sw_wavespvdDisplay.SelectTangentArray = 'None'
sw_wavespvdDisplay.SelectTCoordArray = 'None'
sw_wavespvdDisplay.TextureTransform = 'Transform2'
sw_wavespvdDisplay.OSPRayScaleArray = 'u_sw'
sw_wavespvdDisplay.OSPRayScaleFunction = 'Piecewise Function'
sw_wavespvdDisplay.Assembly = ''
sw_wavespvdDisplay.SelectedBlockSelectors = ['']
sw_wavespvdDisplay.SelectOrientationVectors = 'None'
sw_wavespvdDisplay.ScaleFactor = 0.2999999999999403
sw_wavespvdDisplay.SelectScaleArray = 'u_sw'
sw_wavespvdDisplay.GlyphType = 'Arrow'
sw_wavespvdDisplay.GlyphTableIndexArray = 'u_sw'
sw_wavespvdDisplay.GaussianRadius = 0.014999999999997016
sw_wavespvdDisplay.SetScaleArray = ['POINTS', 'u_sw']
sw_wavespvdDisplay.ScaleTransferFunction = 'Piecewise Function'
sw_wavespvdDisplay.OpacityArray = ['POINTS', 'u_sw']
sw_wavespvdDisplay.OpacityTransferFunction = 'Piecewise Function'
sw_wavespvdDisplay.DataAxesGrid = 'Grid Axes Representation'
sw_wavespvdDisplay.PolarAxes = 'Polar Axes Representation'
sw_wavespvdDisplay.ScalarOpacityFunction = u_swPWF
sw_wavespvdDisplay.ScalarOpacityUnitDistance = 0.28371651140849663
sw_wavespvdDisplay.OpacityArrayName = ['POINTS', 'u_sw']
sw_wavespvdDisplay.SelectInputVectors = [None, '']
sw_wavespvdDisplay.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
sw_wavespvdDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
sw_wavespvdDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for resultLUT in view renderView1
resultLUTColorBar = GetScalarBar(resultLUT, renderView1)
resultLUTColorBar.AutoOrient = 0
resultLUTColorBar.Orientation = 'Horizontal'
resultLUTColorBar.WindowLocation = 'Any Location'
resultLUTColorBar.Position = [0.09745347698334969, 0.2616822429906542]
resultLUTColorBar.Title = 'phi_x'
resultLUTColorBar.ComponentTitle = '(deep water)'
resultLUTColorBar.ScalarBarLength = 0.32999999999999985

# set color bar visibility
resultLUTColorBar.Visibility = 1

# get color legend/bar for u_swLUT in view renderView1
u_swLUTColorBar = GetScalarBar(u_swLUT, renderView1)
u_swLUTColorBar.AutoOrient = 0
u_swLUTColorBar.Orientation = 'Horizontal'
u_swLUTColorBar.WindowLocation = 'Any Location'
u_swLUTColorBar.Position = [0.5193157968378341, 0.2590829457078025]
u_swLUTColorBar.Title = 'u'
u_swLUTColorBar.ComponentTitle = '(shallow water)'
u_swLUTColorBar.ScalarBarLength = 0.3300000000000003

# set color bar visibility
u_swLUTColorBar.Visibility = 1

# show color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
sw_wavespvdDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation scene

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = renderView1
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 16.17999999999679
animationScene1.EndTime = 124.72000000023147
animationScene1.PlayMode = 'Snap To TimeSteps'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(warpByVector1)
# ----------------------------------------------------------------


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------