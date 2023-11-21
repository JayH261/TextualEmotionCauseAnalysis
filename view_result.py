from utils.utils import *

if __name__ == '__main__':
  filepath = "results/allAdamW3e-5_epoch50_stopearly.pkl"
  pretrained_result = read_b(filepath)
  test_metric_ec, test_metric_e, test_metric_c = pretrained_result['test_ecp'], pretrained_result['test_emo'], pretrained_result['test_cau']
    
  print('===== Test Data Average =====')
  print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(test_metric_ec[2]), float_n(test_metric_ec[0]), float_n(test_metric_ec[1])))
  print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(test_metric_e[2]), float_n(test_metric_e[0]), float_n(test_metric_e[1])))
  print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(test_metric_c[2]), float_n(test_metric_c[0]), float_n(test_metric_c[1])))
