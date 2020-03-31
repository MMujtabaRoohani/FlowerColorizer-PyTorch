//
//  ViewController.swift
//  MLProject
//
//  Created by Waqas on 06/12/2018.
//  Copyright © 2018 Mujtaba. All rights reserved.
//

import UIKit
import ALCameraViewController
import Alamofire


class ViewController: UIViewController {

    @IBOutlet weak var imageView: UIImageView!
    let requestURL = "http://10.20.3.186:5000/api"
    var image : UIImage? {
        didSet {
            self.imageView.image = image
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }


    @IBAction func selectImage(_ sender: UIButton!) {
        
        let cameraViewController = CameraViewController.init(croppingParameters: CroppingParameters.init(isEnabled: true, allowResizing: false, allowMoving: true, minimumSize: CGSize.init(width: 100, height: 100)), allowsLibraryAccess: true, allowsSwapCameraOrientation: true, allowVolumeButtonCapture: true) { [weak self] image, asset in
            // Do something with your image here.
            if let img = image {
                self?.image = img
            }
            self?.dismiss(animated: true, completion: nil)
        }
    
        present(cameraViewController, animated: true, completion: nil)

    }
    
    
    @IBAction func didTapNextBtn(_ sender: UIButton) {
        if self.image != nil {
            recolorizePhoto(image: self.image!) { (flag, colouredImage) in
                if !flag {
                    let alert = UIAlertController.init(title: "Error", message: "There was some error connecting to the server", preferredStyle: .alert)
                    alert.addAction(UIAlertAction.init(title: "Okay", style: .default, handler: nil))
                    self.present(alert, animated: true, completion: nil)
                    return
                }
                self.performSegue(withIdentifier: "next", sender: colouredImage!)
            }
        }
        else {
            let alert = UIAlertController.init(title: "Error", message: "Please select any image to continue.", preferredStyle: .alert)
            alert.addAction(UIAlertAction.init(title: "OK", style: .default, handler: nil))
            self.present(alert, animated: true, completion: nil)
        }
    }
    func recolorizePhoto(image: UIImage, callback: @escaping ((Bool, UIImage?) -> Void)){
        let params : [String:Any] = [:]
        
        
        Alamofire.upload(multipartFormData: { (multipart) in
            multipart.append(UIImageJPEGRepresentation(image, 0.6)!, withName: "assets", fileName: "abc.jpg", mimeType: "JPG")
            for (key,value) in params {
                multipart.append("\(value)".data(using: String.Encoding.utf8, allowLossyConversion: false)!, withName: key)
            }
//            if let id = defaultSettings.value(forKey: deviceIdKey) as? String {
//                multipart.append(id.data(using: String.Encoding.utf8, allowLossyConversion: false)!, withName: "deviceId")
//            }
            
        }, to: "\(requestURL)/importer", method: .post) { (result) in
            switch result {
                
                
            case .failure(let error):
                print(error.localizedDescription)
                callback(false, nil)
                
            case .success(request: let request, _, _):
                
                
                request.responseData(completionHandler: { (res) in
                    if res.error != nil {
                        print(#function, res.error?.localizedDescription as Any)
                        callback(false, nil)
                        return
                    }
                    guard let data = res.data else {
                        callback(false, nil)
                        return
                    }
                    if let image = UIImage(data: data) {
                        callback(true, image)
                    }
                    else {
                        callback(false, nil)
                    }
                })
            }
            
            
        }
    }
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "next" {
            
            let dest = segue.destination as! ApplyingMLViewController
            dest.inputImage = self.image
            dest.outputImage = sender as! UIImage
            
        }
    }
    
}

