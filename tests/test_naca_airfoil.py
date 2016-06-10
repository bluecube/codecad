import codecad

def check_parameters(param_obj, thickness, max_camber, max_camber_position):
    codecad.util.check_close(param_obj.thickness, thickness, 0.001)
    codecad.util.check_close(param_obj.max_camber, max_camber, 0.001)
    codecad.util.check_close(param_obj.max_camber_position, max_camber_position, 0.001)

def test_parameters():
    check_parameters(codecad.naca_airfoil.NacaAirfoil("2412"), 0.12, 0.02, 0.4)
    check_parameters(codecad.naca_airfoil.NacaAirfoil("0015"), 0.15, 0.0, 0.0)

