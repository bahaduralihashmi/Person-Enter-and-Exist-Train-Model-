tflite_model_path = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\person_tinycnn_int8.tflite"
cc_file_path = "C:/Users/bahad/OneDrive/Desktop/ultralytics/person_tinycnn_int8.cc"

with open(tflite_model_path, "rb") as f:
    data = f.read()

with open(cc_file_path, "w") as f:
    f.write("#include <cstdint>\n\n")
    f.write(f"const unsigned char waste_classifier_int8_tflite[] = {{\n")
    for i, byte in enumerate(data):
        f.write(f"0x{byte:02x},")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int waste_classifier_int8_tflite_len = {len(data)};\n")
