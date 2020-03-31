//
//  ApplyingMLViewController.swift
//  MLProject
//
//  Created by Waqas Hussain on 06/12/2018.
//  Copyright © 2018 Mujtaba. All rights reserved.
//

import UIKit

class ApplyingMLViewController: UIViewController {

    @IBOutlet weak var inputImageView: UIImageView!
    @IBOutlet weak var outputImageView: UIImageView!
    
    var inputImage: UIImage!
    var outputImage: UIImage!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.inputImageView.image = inputImage
        self.outputImageView.image = outputImage

        // Do any additional setup after loading the view.
    }
    

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
