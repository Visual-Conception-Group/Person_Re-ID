import {Injectable} from "@angular/core"
import {HttpClient} from "@angular/common/http"
@Injectable()
export class VideoProcessService{
    httpurl : String = "http://reid.iiitd.edu.in:80";
    public constructor(private __httpclient : HttpClient){}

    public upload_video(video_file : any)
    {
        return this.__httpclient.post(this.httpurl+"/api/uploadvideo",
        video_file,
        {
            reportProgress : true,
            observe :'events',
            
            
        })
    }

    public upload_video_v2(video_file : any)
    {
        // console.log(video_file)
        return this.__httpclient.post(this.httpurl+"/prid/api/v2/uploadvideo",
        video_file,
        {
            reportProgress : true,
            observe :'events',
            
            
        })
    }

    public get_video_preview(username : any){
        return this.__httpclient.post(this.httpurl+ "/prid/api/v1/getvideopreview",{"username" : username})
    }

    public get_video_preview_v2(username : any){
        return this.__httpclient.post(this.httpurl+ "/prid/api/v2/getvideopreview",{"username" : username})
    }

    public get_frame_count(username:any){
        return this.__httpclient.post(this.httpurl+"/prid/api/v1/getframecount",
        {
            "username" : username
        })
    }

    public get_frame_count_v2(username:any){
        return this.__httpclient.post(this.httpurl+"/prid/api/v2/getframecount",
        {
            "username" : username
        })
    }

    public get_cached_frame_count(username:any){
        return this.__httpclient.post(this.httpurl+"/prid/api/v2/getframecount_demo",
        {
            "username" : username
        })
    }
    
    public get_second_info_api(username:any,second_name:any){
        return this.__httpclient.post(this.httpurl+"/prid/api/v1/getsecondinfo",{
            "username" : username,
            "second_name" : second_name
        })
    }

    public get_second_info_api_v2(username:any,second_name:any){
        return this.__httpclient.post(this.httpurl+"/prid/api/v2/getsecondinfo",{
            "username" : username,
            "second_name" : second_name
        })
    }

    public start_prid_api(formdata:any)
    {
        return this.__httpclient.post(this.httpurl+"/prid/api/v1/prid",formdata);
    }

    public start_prid_api_v2(formdata:any)
    {
        return this.__httpclient.post(this.httpurl+"/prid/api/v2/prid",formdata);
    }

    public get_status_api(username:any){
        return this.__httpclient.post(this.httpurl+"/prid/api/v1/getstatus",{
            "username" : username
        });
    }
    public analysis_done(username:any){
        return this.__httpclient.post(this.httpurl+"/prid/api/v1/analysisdone",{"username":username});
    }

    public prepare_demo(username:any){
        return this.__httpclient.post(this.httpurl+"/prid/api/v1/prepare_demo",{"username":username});
    }

}